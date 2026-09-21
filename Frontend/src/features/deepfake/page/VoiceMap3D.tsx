import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import {
  Billboard,
  Line,
  OrbitControls,
  Sparkles,
  Stars,
} from "@react-three/drei";
import * as THREE from "three";
import type { EmbeddingRecording } from "../types";
import { nearest, normalise } from "./mapGeometry";
import type { Palette } from "./palette";
import { useDfTheme } from "./theme";
import type { MapPointerEvent } from "./VoiceMap2D";

interface VoiceMap3DProps {
  recordings: EmbeddingRecording[];
  coordinates: number[][];
  selectedId: string;
  hoveredId: string | null;
  onHover: (event: MapPointerEvent | null) => void;
  onSelect: (recordingId: string) => void;
}

// React context does not cross react-three-fiber's <Canvas>, so the scene's
// colours arrive as a prop from the component that renders the canvas.
type SceneProps = VoiceMap3DProps & { palette: Palette };

const RADIUS = 0.034;
const SPREAD = 2.1;
/** Each sphere sits inside a slightly larger back-faced shell in the canvas
 *  colour, so touching neighbours read as separate balls. */
const OUTLINE = 1.28;

/** The recordings as an instanced cloud of spheres: one draw call for all of
 *  them, each easing from the centre to its place and growing when chosen. */
const Cloud = ({
  recordings,
  coordinates,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
  palette,
}: SceneProps) => {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const outline = useRef<THREE.InstancedMesh>(null);
  const targets = useMemo(
    () =>
      normalise(coordinates, 3).map(([x, y, z]) =>
        new THREE.Vector3(x, y, z).multiplyScalar(SPREAD),
      ),
    [coordinates],
  );
  const current = useRef<THREE.Vector3[]>([]);
  const scales = useRef<number[]>([]);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    // Keep positions already on screen so a new projection glides; new points start at the centre.
    current.current = targets.map(
      (_, index) => current.current[index] ?? new THREE.Vector3(),
    );
    scales.current = targets.map((_, index) => scales.current[index] ?? 0);
    const instance = mesh.current;
    if (!instance) return;
    recordings.forEach((recording, index) =>
      instance.setColorAt(
        index,
        new THREE.Color(palette.scoreColor(recording.spoof_probability)),
      ),
    );
    if (instance.instanceColor) instance.instanceColor.needsUpdate = true;
  }, [targets, recordings, palette]);

  useFrame((_, delta) => {
    const instance = mesh.current;
    if (!instance) return;
    const ease = 1 - Math.pow(0.02, delta);
    targets.forEach((target, index) => {
      const position = current.current[index];
      if (!position) return;
      position.lerp(target, ease * (0.35 + (index % 7) * 0.05));
      const id = recordings[index]?.recording_id;
      const goal = id === selectedId ? 2.8 : id === hoveredId ? 1.8 : 1;
      scales.current[index] +=
        (goal - scales.current[index]) * Math.min(1, delta * 10);
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
    onHover({
      recordingId: recordings[index].recording_id,
      clientX: event.nativeEvent.clientX,
      clientY: event.nativeEvent.clientY,
    });
  };

  return (
    <>
      <instancedMesh
        ref={outline}
        args={[undefined, undefined, recordings.length]}
        raycast={() => undefined}
      >
        <sphereGeometry args={[RADIUS * OUTLINE, 16, 16]} />
        <meshBasicMaterial color={palette.CANVAS} side={THREE.BackSide} />
      </instancedMesh>
      <instancedMesh
        ref={mesh}
        args={[undefined, undefined, recordings.length]}
        onPointerMove={pointerAt}
        onPointerOut={() => onHover(null)}
        onClick={(event) => {
          if (event.instanceId === undefined) return;
          event.stopPropagation();
          onSelect(recordings[event.instanceId].recording_id);
        }}
      >
        <sphereGeometry args={[RADIUS, 20, 20]} />
        <meshStandardMaterial
          roughness={0.45}
          metalness={0}
          toneMapped={false}
        />
      </instancedMesh>
    </>
  );
};

/** Pulsing halo that always faces the camera, plus lines to the neighbours. */
const SelectionHalo = ({
  position,
  colour,
  ink,
  neighbours,
}: {
  position: THREE.Vector3;
  colour: string;
  ink: string;
  neighbours: THREE.Vector3[];
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
          <meshBasicMaterial color={colour} transparent />
        </mesh>
        <mesh>
          <ringGeometry args={[RADIUS * 3.8, RADIUS * 4.2, 48]} />
          <meshBasicMaterial color={ink} />
        </mesh>
      </Billboard>
      {neighbours.map((neighbour, index) => (
        <Line
          key={index}
          points={[position, neighbour]}
          color={colour}
          lineWidth={1.2}
          dashed
          dashSize={0.03}
          gapSize={0.03}
          transparent
          opacity={0.7}
        />
      ))}
    </group>
  );
};

/** A static dashed-look ring marking one of the visitor's own clips. */
const UserClipMarker = ({
  position,
  ink,
}: {
  position: THREE.Vector3;
  ink: string;
}) => (
  <Billboard position={position}>
    <mesh>
      <ringGeometry args={[RADIUS * 2.2, RADIUS * 2.6, 6]} />
      <meshBasicMaterial color={ink} transparent opacity={0.9} />
    </mesh>
  </Billboard>
);

/**
 * The 3D voice map, drawn with react-three-fiber. It auto-rotates slowly
 * until someone grabs it, and hover/click behave exactly like the 2D map.
 */
const VoiceMap3D = (props: VoiceMap3DProps) => {
  const { recordings, coordinates, selectedId } = props;
  const [interacted, setInteracted] = useState(false);
  const { theme, palette } = useDfTheme();
  const light = theme === "light";
  // In daylight the scene is a pale "well" with violet dust instead of a night sky.
  const background = light ? "#eef0f4" : palette.CANVAS;
  const positions = useMemo(
    () =>
      normalise(coordinates, 3).map(([x, y, z]) =>
        new THREE.Vector3(x, y, z).multiplyScalar(SPREAD),
      ),
    [coordinates],
  );
  const selectedIndex = recordings.findIndex(
    (recording) => recording.recording_id === selectedId,
  );
  const neighbours = useMemo(() => {
    if (selectedIndex < 0) return [];
    const raw = positions.map((vector) => vector.toArray());
    return nearest(raw, selectedIndex, 5).map((index) => positions[index]);
  }, [positions, selectedIndex]);

  return (
    <Canvas
      camera={{ position: [0, 0.3, 3.6], fov: 50 }}
      dpr={[1, 2]}
      onPointerMissed={() => props.onHover(null)}
    >
      <color attach="background" args={[background]} />
      <fog attach="fog" args={[background, 4, 9]} />
      {/* neutral white light, so each ball shows its own score colour at full strength */}
      <ambientLight intensity={light ? 2.2 : 2} />
      <directionalLight position={[3, 4, 5]} intensity={light ? 1.2 : 1.6} />
      <pointLight position={[-3, -2, 2]} intensity={10} color="#ffffff" />
      {light ? (
        <Sparkles
          count={160}
          scale={7}
          size={2.5}
          speed={0.25}
          opacity={0.5}
          color={palette.MID}
        />
      ) : (
        <Stars
          radius={30}
          depth={20}
          count={1500}
          factor={3}
          fade
          speed={0.6}
        />
      )}
      <Cloud {...props} palette={palette} />
      {recordings.map((recording, index) =>
        recording.uploaded && positions[index] ? (
          <UserClipMarker
            key={recording.recording_id}
            position={positions[index]}
            ink={palette.INK}
          />
        ) : null,
      )}
      {selectedIndex >= 0 && positions[selectedIndex] && (
        <SelectionHalo
          position={positions[selectedIndex]}
          colour={palette.scoreColor(
            recordings[selectedIndex].spoof_probability,
          )}
          ink={palette.INK}
          neighbours={neighbours}
        />
      )}
      <OrbitControls
        enableDamping
        autoRotate={!interacted && !props.hoveredId}
        autoRotateSpeed={0.7}
        minDistance={1.5}
        maxDistance={8}
        onStart={() => setInteracted(true)}
      />
    </Canvas>
  );
};

export default VoiceMap3D;
