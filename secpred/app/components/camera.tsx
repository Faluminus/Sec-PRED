import { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame,useThree} from '@react-three/fiber';
import { PerspectiveCamera,} from '@react-three/drei';
import { xor } from 'three/examples/jsm/nodes/Nodes.js';

export default function MyCamera() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (event:MouseEvent) => {
      setMousePosition({ x: event.clientX, y: event.clientY });
    };

    // Add the event listener
    window.addEventListener('mousemove', handleMouseMove);

    // Clean up the event listener on component unmount
  })
  
  const { camera } = useThree();
  useFrame(()=>{
    
    camera.rotation.x = mousePosition['x'] * 0.3
    camera.rotation.y = mousePosition['y'] * 0.3
  })
  console.log(mousePosition['x'])

  return null
}