import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Billboard, Html, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { EmbeddingRecording } from "../types";
import { declutter, nearest, normalise } from "./mapGeometry";
import type { Palette } from "./palette";
import { usePalette } from "./theme";
import type { MapPointerEvent } from "./VoiceMap2D";

interface VoiceMap3DProps {
  recordings: EmbeddingRecording[];
  coordinates: number[][];
  selectedId: string;
  hoveredId: string | null;
  onHover: (event: MapPointerEvent | null) => void;
  onSelect: (recordingId: string) => void;
  /** Nudge overlapping points apart so every clip can be seen. */
  spread?: boolean;
}

// React context does not cross react-three-fiber's <Canvas>, so the scene's
// colours arrive as a prop from the component that renders the canvas.
type SceneProps = VoiceMap3DProps & { palette: Palette };

/** Half the side of the reference box the cloud is scaled into. */
const SPREAD = 1.6;
const FLOOR = -SPREAD * 1.08;
const CAMERA: [number, number, number] = [3.1, 2.1, 4.3];
/** Camera distance at which a point is drawn at its nominal pixel size. */
const REFERENCE_DISTANCE = Math.hypot(...CAMERA);
/** Point diameters in CSS pixels, at the reference distance. */
const SIZE = { rest: 9, hovered: 15, selected: 19 };
/** The selected point's own layers, in pixels: a white gap, then an ink ring. */
const SPOT = { gap: 2.5, ring: 2.5 };
/** Extra pixels of slack around a point when picking it with the mouse. */
const PICK_SLACK = 4;
/** The pale "well" the 2D map also sits on. */
const BACKGROUND = "#eef0f4";

/**
 * Points are flat, screen-facing discs with a thin white rim — the look of
 * TensorFlow's Embedding Projector and most published embedding plots — not
 * lit spheres, so crowded clusters stay a field of countable dots instead of
 * merging into one shaded mass. Depth is carried by redundant cues: size
 * falls off with distance, far points fade toward the background, and every
 * point casts a faint shadow onto a floor grid.
 */
const VERTEX = /* glsl */ `
  attribute float size;
  attribute vec3 color;
  uniform float pixelRatio;
  uniform float referenceDistance;
  uniform float fogNear;
  uniform float fogFar;
  uniform float maxSize;
  varying vec3 vColor;
  varying float vFade;
  varying float vSize;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float depth = -mv.z;
    vSize = clamp(size * pow(referenceDistance / depth, 0.5), 3.0, maxSize) * pixelRatio;
    gl_PointSize = vSize;
    vColor = color;
    vFade = smoothstep(fogNear, fogFar, depth);
    gl_Position = projectionMatrix * mv;
  }
`;

const FRAGMENT = /* glsl */ `
  uniform vec3 rim;
  uniform vec3 gapColour;
  uniform vec3 background;
  uniform float rimPixels;
  uniform float gapPixels;
  uniform float fadeStrength;
  uniform float opacity;
  varying vec3 vColor;
  varying float vFade;
  varying float vSize;
  void main() {
    // distance from the disc centre in pixels
    float r = length(gl_PointCoord - 0.5) * vSize;
    float edge = vSize * 0.5;
    if (r > edge) discard;
    float aa = 1.0;
    // from the outside in: rim, optional gap, body
    float inRim = smoothstep(edge - rimPixels - aa, edge - rimPixels, r);
    float inGap = smoothstep(edge - rimPixels - gapPixels - aa, edge - rimPixels - gapPixels, r);
    vec3 colour = mix(vColor, gapColour, inGap);
    colour = mix(colour, rim, inRim);
    colour = mix(colour, background, vFade * fadeStrength);
    float alpha = (1.0 - smoothstep(edge - aa, edge, r)) * opacity;
    gl_FragColor = vec4(colour, alpha);
    #include <colorspace_fragment>
  }
`;

const pointMaterial = (
  rimColour: string,
  rimPixels: number,
  opacity: number,
  { gapPixels = 0, fadeStrength = 0.35, maxSize = 24 } = {},
) =>
  new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    // Points are sorted back-to-front every frame, so blending alone orders them.
    depthWrite: false,
    depthTest: false,
    uniforms: {
      pixelRatio: { value: 1 },
      referenceDistance: { value: REFERENCE_DISTANCE },
      fogNear: { value: REFERENCE_DISTANCE - SPREAD },
      fogFar: { value: REFERENCE_DISTANCE + SPREAD * 1.6 },
      rim: { value: new THREE.Color(rimColour) },
      background: { value: new THREE.Color(BACKGROUND) },
      rimPixels: { value: rimPixels },
      gapColour: { value: new THREE.Color("#ffffff") },
      gapPixels: { value: gapPixels },
      fadeStrength: { value: fadeStrength },
      maxSize: { value: maxSize },
      opacity: { value: opacity },
    },
  });

/** Minimum gap between point centres, in world units, when spreading overlaps. */
const MIN_GAP = 0.075;

const toTargets = (coordinates: number[][], spread = false) => {
  const world = normalise(coordinates, 3).map((row) => row.map((value) => value * SPREAD));
  return (spread ? declutter(world, MIN_GAP) : world).map(([x, y, z]) => new THREE.Vector3(x, y, z));
};

/** The recordings as one point cloud plus its floor shadows, easing from the
 *  centre to their places, with mouse picking done in screen space. */
const Cloud = ({ recordings, coordinates, selectedId, hoveredId, onHover, onSelect, palette, spread }: SceneProps) => {
  const { camera, gl, size: viewport } = useThree();
  const count = recordings.length;
  const targets = useMemo(() => toTargets(coordinates, spread), [coordinates, spread]);
  const current = useRef<THREE.Vector3[]>([]);
  const sizes = useRef<number[]>([]);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    g.setAttribute("size", new THREE.BufferAttribute(new Float32Array(count), 1));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(count), 1));
    return g;
  }, [count]);
  const shadowGeometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    g.setAttribute("size", new THREE.BufferAttribute(new Float32Array(count).fill(5), 1));
    return g;
  }, [count]);
  const material = useMemo(() => pointMaterial("#ffffff", 1.4, 0.95), []);
  // The selected clip, drawn once more above everything: its own colour,
  // wrapped in a white gap and an ink ring, never faded by depth.
  const spotGeometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3), 3));
    g.setAttribute("size", new THREE.BufferAttribute(new Float32Array([SIZE.selected + 2 * (SPOT.gap + SPOT.ring)]), 1));
    return g;
  }, []);
  const spotMaterial = useMemo(
    () => pointMaterial(palette.INK, SPOT.ring, 1, { gapPixels: SPOT.gap, fadeStrength: 0, maxSize: 40 }),
    [palette],
  );
  const beacon = useRef<THREE.Group>(null);
  const selectedIndex = recordings.findIndex((recording) => recording.recording_id === selectedId);
  const shadowMaterial = useMemo(() => {
    const m = pointMaterial(palette.MID, 0, 0.22);
    m.depthTest = true;
    return m;
  }, [palette]);
  useEffect(() => () => {
    geometry.dispose();
    shadowGeometry.dispose();
  }, [geometry, shadowGeometry]);
  useEffect(() => () => {
    material.dispose();
    shadowMaterial.dispose();
    spotMaterial.dispose();
  }, [material, shadowMaterial, spotMaterial]);
  useEffect(() => () => spotGeometry.dispose(), [spotGeometry]);

  useEffect(() => {
    if (selectedIndex < 0) return;
    const colour = new THREE.Color(palette.scoreColor(recordings[selectedIndex].spoof_probability));
    const attr = spotGeometry.getAttribute("color") as THREE.BufferAttribute;
    attr.setXYZ(0, colour.r, colour.g, colour.b);
    attr.needsUpdate = true;
  }, [selectedIndex, recordings, palette, spotGeometry]);

  useEffect(() => {
    // Keep positions already on screen so a new projection glides; new points start at the centre.
    current.current = targets.map((_, index) => current.current[index] ?? new THREE.Vector3());
    sizes.current = targets.map((_, index) => sizes.current[index] ?? 0);
    const colours = geometry.getAttribute("color") as THREE.BufferAttribute;
    const shadowColours = shadowGeometry.getAttribute("color") as THREE.BufferAttribute;
    const shade = new THREE.Color(palette.MID);
    recordings.forEach((recording, index) => {
      const colour = new THREE.Color(palette.scoreColor(recording.spoof_probability));
      colours.setXYZ(index, colour.r, colour.g, colour.b);
      shadowColours.setXYZ(index, shade.r, shade.g, shade.b);
    });
    colours.needsUpdate = true;
    shadowColours.needsUpdate = true;
  }, [targets, recordings, palette, geometry, shadowGeometry]);

  const order = useRef<number[]>([]);
  const depth = useMemo(() => new THREE.Vector3(), []);

  useFrame((_, delta) => {
    const ease = 1 - Math.pow(0.02, delta);
    const positions = geometry.getAttribute("position") as THREE.BufferAttribute;
    const pointSizes = geometry.getAttribute("size") as THREE.BufferAttribute;
    const shadows = shadowGeometry.getAttribute("position") as THREE.BufferAttribute;
    targets.forEach((target, index) => {
      const position = current.current[index];
      if (!position) return;
      position.lerp(target, ease * (0.35 + (index % 7) * 0.05));
      const id = recordings[index]?.recording_id;
      const goal = id === selectedId ? SIZE.selected : id === hoveredId ? SIZE.hovered : SIZE.rest;
      sizes.current[index] += (goal - sizes.current[index]) * Math.min(1, delta * 12);
      positions.setXYZ(index, position.x, position.y, position.z);
      pointSizes.setX(index, sizes.current[index]);
      shadows.setXYZ(index, position.x, FLOOR, position.z);
    });
    positions.needsUpdate = true;
    pointSizes.needsUpdate = true;
    shadows.needsUpdate = true;

    const chosen = current.current[selectedIndex];
    if (chosen) {
      (spotGeometry.getAttribute("position") as THREE.BufferAttribute).setXYZ(0, chosen.x, chosen.y, chosen.z);
      spotGeometry.getAttribute("position").needsUpdate = true;
      beacon.current?.position.copy(chosen);
    }

    // Far points first, then the hovered and selected ones on top of everything.
    const rank = (index: number) => {
      const id = recordings[index]?.recording_id;
      if (id === selectedId) return 1e6;
      if (id === hoveredId) return 1e5;
      const position = current.current[index];
      return position ? -depth.copy(position).distanceToSquared(camera.position) : 0;
    };
    const keyed = recordings.map((_, index) => [rank(index), index] as const);
    keyed.sort((a, b) => a[0] - b[0]);
    order.current = keyed.map(([, index]) => index);
    const indexAttr = geometry.getIndex()!;
    order.current.forEach((pointIndex, slot) => indexAttr.setX(slot, pointIndex));
    indexAttr.needsUpdate = true;

    material.uniforms.pixelRatio.value = gl.getPixelRatio();
    shadowMaterial.uniforms.pixelRatio.value = gl.getPixelRatio();
    spotMaterial.uniforms.pixelRatio.value = gl.getPixelRatio();
  });

  // Screen-space picking: the point whose drawn disc is under the cursor,
  // preferring the one in front. Tiny points stay easy to hit.
  const pick = (clientX: number, clientY: number): number | null => {
    const box = gl.domElement.getBoundingClientRect();
    const mx = clientX - box.left;
    const my = clientY - box.top;
    const projected = new THREE.Vector3();
    let best: number | null = null;
    let bestScore = Infinity;
    current.current.forEach((position, index) => {
      projected.copy(position).project(camera);
      if (projected.z > 1) return;
      const sx = (projected.x * 0.5 + 0.5) * viewport.width;
      const sy = (-projected.y * 0.5 + 0.5) * viewport.height;
      const distance = Math.hypot(sx - mx, sy - my);
      const cameraDistance = position.distanceTo(camera.position);
      const radius = (sizes.current[index] * REFERENCE_DISTANCE) / cameraDistance / 2 + PICK_SLACK;
      if (distance > radius) return;
      // Closest to the cursor wins; depth breaks near-ties.
      const score = distance + cameraDistance * 0.5;
      if (score < bestScore) {
        bestScore = score;
        best = index;
      }
    });
    return best;
  };

  const latest = useRef({ recordings, onHover, onSelect, hoveredId });
  latest.current = { recordings, onHover, onSelect, hoveredId };

  useEffect(() => {
    const element = gl.domElement;
    let down: { x: number; y: number } | null = null;
    const move = (event: PointerEvent) => {
      const index = pick(event.clientX, event.clientY);
      const { recordings: list, onHover: hover, hoveredId: hovered } = latest.current;
      element.style.cursor = index === null ? "grab" : "pointer";
      if (index === null) {
        if (hovered) hover(null);
        return;
      }
      hover({ recordingId: list[index].recording_id, clientX: event.clientX, clientY: event.clientY });
    };
    const press = (event: PointerEvent) => {
      down = { x: event.clientX, y: event.clientY };
    };
    const release = (event: PointerEvent) => {
      // A drag rotates the view; only a still click selects.
      const start = down;
      down = null;
      if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) return;
      const index = pick(event.clientX, event.clientY);
      if (index !== null) latest.current.onSelect(latest.current.recordings[index].recording_id);
    };
    const leave = () => latest.current.onHover(null);
    element.addEventListener("pointermove", move);
    element.addEventListener("pointerdown", press);
    element.addEventListener("pointerup", release);
    element.addEventListener("pointerleave", leave);
    return () => {
      element.removeEventListener("pointermove", move);
      element.removeEventListener("pointerdown", press);
      element.removeEventListener("pointerup", release);
      element.removeEventListener("pointerleave", leave);
    };
    // pick reads refs and the live camera, so it never goes stale
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, camera, viewport]);

  return (
    <>
      <points geometry={shadowGeometry} material={shadowMaterial} renderOrder={1} raycast={() => undefined} />
      <points geometry={geometry} material={material} renderOrder={10} raycast={() => undefined} />
      {selectedIndex >= 0 && (
        <>
          <points geometry={spotGeometry} material={spotMaterial} renderOrder={30} raycast={() => undefined} />
          {/* a screen-sized pulsing beacon that reads through any crowd */}
          <group ref={beacon}>
            <Html center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
              <span
                className="df-beacon"
                aria-hidden
                style={{ ["--df-beacon" as string]: palette.scoreColor(recordings[selectedIndex].spoof_probability) }}
              />
            </Html>
          </group>
        </>
      )}
    </>
  );
};

/** A faint floor grid and wireframe box: a frame of reference for the eye,
 *  deliberately unlabelled because the projected axes have no units. */
const ReferenceFrame = ({ palette }: { palette: Palette }) => {
  const box = useMemo(
    () => new THREE.EdgesGeometry(new THREE.BoxGeometry(SPREAD * 2.16, SPREAD * 2.16, SPREAD * 2.16)),
    [],
  );
  useEffect(() => () => box.dispose(), [box]);
  return (
    <group>
      <gridHelper
        args={[SPREAD * 2.16, 12, palette.ink(0.22), palette.ink(0.08)]}
        position={[0, FLOOR, 0]}
      />
      <lineSegments geometry={box}>
        <lineBasicMaterial color={palette.INK} transparent opacity={0.1} />
      </lineSegments>
    </group>
  );
};

/** The selection's context: a drop line to the floor and dashed lines to
 *  its five nearest neighbours, drawn above the cloud. The point
 *  itself and its beacon are drawn by the cloud. */
const SelectionLinks = ({
  position,
  ink,
  neighbours,
}: {
  position: THREE.Vector3;
  ink: string;
  neighbours: { position: THREE.Vector3; colour: string }[];
}) => (
  <group>
    <Line
      points={[position, new THREE.Vector3(position.x, FLOOR, position.z)]}
      color={ink}
      lineWidth={1}
      transparent
      opacity={0.35}
    />
    {neighbours.map((neighbour, index) => (
      <Line
        key={index}
        points={[position, neighbour.position]}
        color={neighbour.colour}
        lineWidth={1.4}
        dashed
        dashSize={0.035}
        gapSize={0.03}
        transparent
        opacity={0.85}
        depthTest={false}
        renderOrder={20}
      />
    ))}
  </group>
);

/** A hexagonal ring marking one of the visitor's own clips. */
const UserClipMarker = ({ position, ink }: { position: THREE.Vector3; ink: string }) => (
  <Billboard position={position}>
    <mesh renderOrder={15}>
      <ringGeometry args={[0.05, 0.058, 6]} />
      <meshBasicMaterial color={ink} transparent opacity={0.9} depthTest={false} />
    </mesh>
  </Billboard>
);

/**
 * The 3D voice map, drawn with react-three-fiber. It auto-rotates slowly
 * until someone grabs it, and hover/click behave exactly like the 2D map.
 */
const VoiceMap3D = (props: VoiceMap3DProps) => {
  const { recordings, coordinates, selectedId, spread = true } = props;
  const [interacted, setInteracted] = useState(false);
  const palette = usePalette();
  const positions = useMemo(() => toTargets(coordinates, spread), [coordinates, spread]);
  // Neighbours come from the projection itself, not the nudged display.
  const projected = useMemo(() => toTargets(coordinates), [coordinates]);
  const selectedIndex = recordings.findIndex((recording) => recording.recording_id === selectedId);
  const neighbourIndices = useMemo(
    () => (selectedIndex < 0 ? [] : nearest(projected.map((vector) => vector.toArray()), selectedIndex, 5)),
    [projected, selectedIndex],
  );
  const neighbours = neighbourIndices.map((index) => ({
    position: positions[index],
    colour: palette.scoreColor(recordings[index].spoof_probability),
  }));

  return (
    <Canvas camera={{ position: CAMERA, fov: 42 }} dpr={[1, 2]} style={{ cursor: "grab" }}>
      <color attach="background" args={[BACKGROUND]} />
      <ReferenceFrame palette={palette} />
      <Cloud {...props} spread={spread} palette={palette} />
      {recordings.map((recording, index) =>
        recording.uploaded && positions[index] ? (
          <UserClipMarker key={recording.recording_id} position={positions[index]} ink={palette.INK} />
        ) : null,
      )}
      {selectedIndex >= 0 && positions[selectedIndex] && (
        <SelectionLinks position={positions[selectedIndex]} ink={palette.INK} neighbours={neighbours} />
      )}
      <OrbitControls
        enableDamping
        autoRotate={!interacted && !props.hoveredId}
        autoRotateSpeed={0.5}
        minDistance={1.8}
        maxDistance={10}
        onStart={() => setInteracted(true)}
      />
    </Canvas>
  );
};

export default VoiceMap3D;
