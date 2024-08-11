import React, { useEffect, useRef, useState } from 'react'
import { Canvas, useFrame,useLoader } from '@react-three/fiber'
import { OrbitControls,useGLTF,Html} from '@react-three/drei'




export default function Box(props:any){

  const { gltf_path, ...otherProps } = props;

  const ref = useRef<any>(null);
  const gltf = useGLTF(gltf_path);

  
  
  return (
    <mesh
      castShadow
      {...otherProps}
      ref={ref}>
      <primitive object={gltf.scene.clone()} />
    </mesh>
  )
}

