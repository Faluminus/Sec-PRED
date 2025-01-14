import React, { useRef, useMemo,useEffect } from 'react';
import { useGLTF } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

export default function CombinedMesh() {
  const groupRef = useRef<THREE.Group>(null);

  // Load GLTF models once
  const gltf1ft4 = useGLTF('/proteins/1ft4_green.gltf');
  const gltf2ace = useGLTF('/proteins/2ace_purple.gltf');
  const gltf2atc = useGLTF('/proteins/2atc_blue.gltf');
  const gltf2RIK = useGLTF('/proteins/2RIK_red.gltf');
  const gltf3cg5 = useGLTF('/proteins/3cg5_lightblue.gltf');
  const gltf3cgw = useGLTF('/proteins/3cgw_yellow.gltf');
  const gltf6yl0 = useGLTF('/proteins/6yl0_green.gltf');

  // Define model configurations
  const models = useMemo(() => [
    { gltf: gltf1ft4, position: [-2, 7, -11], scale: [0.1, 0.1, 0.1], rotation: [-0.6, -1.7, -0.9] },
    { gltf: gltf2ace, position: [-2, -9, -10], scale: [0.1, 0.1, 0.1], rotation: [-0.6, -1.7, -0.9] },
    { gltf: gltf2atc, position: [-12, 1, -10], scale: [0.1, 0.1, 0.1], rotation: [-0.6, -1.7, -0.9] },
    { gltf: gltf2RIK, position: [-26, -7, -16], scale: [0.1, 0.1, 0.1], rotation: [1, -0.5, -0.9] },
    { gltf: gltf3cg5, position: [-18, 11, -12], scale: [0.1, 0.1, 0.1], rotation: [1, -0.5, -0.9] },
    { gltf: gltf3cgw, position: [15, 4, -12], scale: [0.1, 0.1, 0.1], rotation: [1, -0.5, -0.9] },
    { gltf: gltf6yl0, position: [-11, -8, -13], scale: [0.1, 0.1, 0.1], rotation: [1, -0.5, -0.9] },
    { gltf: gltf2RIK, position: [27, -5, -15], scale: [0.1, 0.1, 0.1], rotation: [1, -0.5, -0.9] },
    { gltf: gltf2atc, position: [11, 4, -20], scale: [0.1, 0.1, 0.1], rotation: [-0.6, -1.7, -0.9] },
    { gltf: gltf3cg5, position: [11, -9, -12], scale: [0.1, 0.1, 0.1], rotation: [1, -0.5, -0.9] },
    { gltf: gltf2ace, position: [-2, -9, -10], scale: [0.1, 0.1, 0.1], rotation: [-0.6, -1.7, -0.9] },
  ], [gltf1ft4, gltf2ace, gltf2atc, gltf2RIK, gltf3cg5, gltf3cgw, gltf6yl0]);

  return (
    <group ref={groupRef} castShadow>
      {models.map((model, index) => (
        <primitive
          key={index}
          object={model.gltf.scene.clone()}
          position={model.position}
          scale={model.scale}
          rotation={model.rotation}
        />
      ))}
    </group>
  );
}


