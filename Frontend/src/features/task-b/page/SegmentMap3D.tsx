import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { Billboard, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { DiarizationSegment, ProjectionPoint } from "../types";
import type { Palette } from "./palette";
import type { MapPointerEvent } from "./SegmentMap2D";
import { useDzTheme } from "./theme";

interface SegmentMap3DProps {
  points: ProjectionPoint[];
  segmentsById: Map<string, DiarizationSegment>;
  speakers: string[];
  selectedId: string | null;
  hoveredId: string | null;
  /** The selected segment's nearest segments by cosine in embedding space. */
  neighbourIds: string[];
  onHover: (event: MapPointerEvent | null) => void;
  onSelect: (segmentId: string) => void;
}

// React context does not cross react-three-fiber's <Canvas>, so the scene's
// colours arrive as props from the component that renders the canvas.
type SceneProps = SegmentMap3DProps & { palette: Palette; well: string; positions: THREE.Vector3[] };

const RADIUS = 0.04;
/** Radius of the sphere the cloud is scaled into. With the camera at
 *  CAMERA_DISTANCE (fov 50), a point swung all the way towards the camera
 *  still sits inside the frame: (4.2 − 1.3) · tan 25° ≈ 1.35 ≥ 1.3. */
const SPREAD = 1.3;
const CAMERA_DISTANCE = 4.2;
/** Each sphere sits inside a slightly larger back-faced shell in the well
 *  colour, so touching neighbours read as separate balls. */
const OUTLINE = 1.28;
/** How far an uncertain segment's colour is pulled towards the background —
 *  the 3D stand-in for the 2D map's 0.45 fill opacity. Same hue, never a new one. */
const UNCERTAIN_FADE = 0.55;

/** Centre the cloud and scale it with ONE factor for all three axes, so it
 *  keeps its shape (a per-axis stretch would invent structure). The factor
 *  comes from the farthest point's distance, not a per-axis extent, so the
 *  cloud fits one sphere and stays in frame however it is rotated. */
function layout(points: ProjectionPoint[]): THREE.Vector3[] {
  if (points.length === 0) return [];
  const axes = [points.map((p) => p.x), points.map((p) => p.y), points.map((p) => p.z ?? 0)];
  const centre = new THREE.Vector3(
    ...(axes.map((values) => (Math.min(...values) + Math.max(...values)) / 2) as [number, number, number]),
  );
  const offsets = points.map((p) => new THREE.Vector3(p.x, p.y, p.z ?? 0).sub(centre));
  const radius = Math.max(1e-6, ...offsets.map((offset) => offset.length()));
  return offsets.map((offset) => offset.multiplyScalar(SPREAD / radius));
}

/** Every segment as an instanced cloud of spheres: one draw call for all of
 *  them, each easing from the centre to its place and growing when chosen. */
const Cloud = ({ points, segmentsById, speakers, selectedId, hoveredId, onHover, onSelect, palette, well, positions }: SceneProps) => {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const outline = useRef<THREE.InstancedMesh>(null);
  const current = useRef<THREE.Vector3[]>([]);
  const scales = useRef<number[]>([]);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    // Keep positions already on screen so a new layout glides; new points start at the centre.
    current.current = positions.map((_, index) => current.current[index] ?? new THREE.Vector3());
    scales.current = positions.map((_, index) => scales.current[index] ?? 0);
    const instance = mesh.current;
    if (!instance) return;
    const background = new THREE.Color(well);
    points.forEach((point, index) => {
      const colour = new THREE.Color(palette.speakerColor(speakers, point.speaker));
      if (segmentsById.get(point.id)?.confidence_bucket === "uncertain") colour.lerp(background, UNCERTAIN_FADE);
      instance.setColorAt(index, colour);
    });
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true;
  }, [positions, points, segmentsById, speakers, palette, well]);

  useFrame((_, delta) => {
    const instance = mesh.current;
    if (!instance) return;
    const ease = 1 - Math.pow(0.02, delta);
    positions.forEach((target, index) => {
      const position = current.current[index];
      if (!position) return;
      position.lerp(target, ease * (0.35 + (index % 7) * 0.05));
      const id = points[index]?.id;
      const goal = id === selectedId ? 2.8 : id === hoveredId ? 1.8 : 1;
      scales.current[index] += (goal - scales.current[index]) * Math.min(1, delta * 10);
      dummy.position.copy(position);
      dummy.scale.setScalar(scales.current[index]);
      dummy.updateMatrix();
      instance.setMatrixAt(index, dummy.matrix);
      outline.current?.setMatrixAt(index, dummy.matrix);
    });
    instance.instanceMatrix.needsUpdate = true;
    if (outline.current) outline.current.instanceMatrix.needsUpdate = true;
    // Raycasting culls against this sphere; left stale from the fly-in it
    // would be tiny, and hovering the settled points would miss.
    instance.computeBoundingSphere();
  });

  const pointerAt = (event: ThreeEvent<PointerEvent>) => {
    const index = event.instanceId;
    if (index === undefined) return;
    event.stopPropagation();
    onHover({ segmentId: points[index].id, clientX: event.nativeEvent.clientX, clientY: event.nativeEvent.clientY });
  };

  return (
    <>
      <instancedMesh ref={outline} args={[undefined, undefined, points.length]} raycast={() => undefined}>
        <sphereGeometry args={[RADIUS * OUTLINE, 16, 16]} />
        <meshBasicMaterial color={well} side={THREE.BackSide} />
      </instancedMesh>
      <instancedMesh
        ref={mesh}
        args={[undefined, undefined, points.length]}
        onPointerMove={pointerAt}
        onPointerOut={() => onHover(null)}
        onClick={(event) => {
          if (event.instanceId === undefined) return;
          event.stopPropagation();
          onSelect(points[event.instanceId].id);
        }}
      >
        <sphereGeometry args={[RADIUS, 20, 20]} />
        <meshStandardMaterial roughness={0.45} metalness={0} toneMapped={false} />
      </instancedMesh>
    </>
  );
};

/** Pulsing halo that always faces the camera — selection by size and ripple,
 *  as on the 2D map — plus dashed lines to the nearest segments, each in the
 *  neighbour's own speaker colour. */
const SelectionHalo = ({
  position,
  colour,
  ink,
  neighbours,
}: {
  position: THREE.Vector3;
  colour: string;
  ink: string;
  neighbours: { position: THREE.Vector3; colour: string }[];
}) => {
  const ring = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ring.current) return;
    const t = (clock.getElapsedTime() % 1.6) / 1.6;
    ring.current.scale.setScalar(1 + t * 2.2);
    (ring.current.material as THREE.MeshBasicMaterial).opacity = 0.9 * (1 - t);
  });
  return (
    <group>
      <Billboard position={position}>
        <mesh ref={ring}>
          <ringGeometry args={[RADIUS * 3, RADIUS * 3.6, 48]} />
          <meshBasicMaterial color={colour} transparent toneMapped={false} />
        </mesh>
        <mesh>
          <ringGeometry args={[RADIUS * 3.8, RADIUS * 4.2, 48]} />
          <meshBasicMaterial color={ink} toneMapped={false} />
        </mesh>
      </Billboard>
      {neighbours.map((neighbour, index) => (
        <Line
          key={index}
          points={[position, neighbour.position]}
          color={neighbour.colour}
          lineWidth={1.4}
          dashed
          dashSize={0.04}
          gapSize={0.05}
          transparent
          opacity={0.8}
        />
      ))}
    </group>
  );
};

/**
 * The 3D segment map, drawn with react-three-fiber. It auto-rotates slowly
 * until someone grabs it; hover and click behave exactly like the 2D map, and
 * the neighbour lines still come from cosine in the full embedding space.
 */
const SegmentMap3D = (props: SegmentMap3DProps) => {
  const { points, speakers, selectedId, hoveredId, neighbourIds } = props;
  const { theme, palette } = useDzTheme();
  const [interacted, setInteracted] = useState(false);

  // The canvas is transparent over `.dz-well`; the fog and the ball outlines
  // need that same colour, so read it from the CSS rather than duplicate it.
  const wrapper = useRef<HTMLDivElement>(null);
  const [well, setWell] = useState(palette.CANVAS);
  useLayoutEffect(() => {
    const value = wrapper.current ? getComputedStyle(wrapper.current).getPropertyValue("--dz-well").trim() : "";
    setWell(value || palette.CANVAS);
  }, [theme, palette]);

  const positions = useMemo(() => layout(points), [points]);
  const indexById = useMemo(() => new Map(points.map((p, i) => [p.id, i])), [points]);
  const selectedIndex = selectedId !== null ? indexById.get(selectedId) : undefined;
  const neighbours = useMemo(
    () =>
      neighbourIds.flatMap((id) => {
        const index = indexById.get(id);
        return index === undefined
          ? []
          : [{ position: positions[index], colour: palette.speakerColor(speakers, points[index].speaker) }];
      }),
    [neighbourIds, indexById, positions, points, speakers, palette],
  );

  return (
    <div ref={wrapper} className="h-full w-full" data-testid="segment-map-3d">
      <Canvas
        camera={{ position: [0, 0.3, CAMERA_DISTANCE], fov: 50 }}
        dpr={[1, 2]}
        gl={{ alpha: true }}
        aria-label="3D segment map: one ball per segment, coloured by speaker. Drag to rotate, scroll to zoom. Switch to 2D to use the keyboard."
        onPointerMissed={() => props.onHover(null)}
      >
        <fog attach="fog" args={[well, 4, 9]} />
        {/* neutral white light, so each ball shows its speaker colour at full strength */}
        <ambientLight intensity={theme === "light" ? 2.2 : 2} />
        <directionalLight position={[3, 4, 5]} intensity={theme === "light" ? 1.2 : 1.6} />
        <pointLight position={[-3, -2, 2]} intensity={10} color="#ffffff" />
        <Cloud {...props} palette={palette} well={well} positions={positions} />
        {selectedIndex !== undefined && positions[selectedIndex] && (
          <SelectionHalo
            position={positions[selectedIndex]}
            colour={palette.speakerColor(speakers, points[selectedIndex].speaker)}
            ink={palette.INK}
            neighbours={neighbours}
          />
        )}
        <OrbitControls
          enableDamping
          autoRotate={!interacted && !hoveredId}
          autoRotateSpeed={0.7}
          minDistance={1.5}
          maxDistance={8}
          onStart={() => setInteracted(true)}
        />
      </Canvas>
    </div>
  );
};

export default SegmentMap3D;
