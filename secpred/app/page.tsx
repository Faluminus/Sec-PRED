"use client"
import { Canvas,useFrame,PerspectiveCameraProps} from '@react-three/fiber';
import {Html,Clone,SpotLight} from '@react-three/drei';
import CombinedMesh from './components/combinedMesh';
import {useEffect,useState} from 'react';
import { useRouter } from 'next/navigation'

export default function Home() {
  const router = useRouter()
  const [width,setWidth] = useState<number>()
  
  const [height,setHeight] = useState<number>()

  useEffect(() =>{
    setHeight(window.innerHeight)
    setWidth(window.innerWidth)
  })

  const handleClick = () => {
    router.push('/predictionPage');
  };
  

  return (
      <>
        <Canvas shadows frameloop="demand">
        <ambientLight intensity={0.6} castShadow />
        <spotLight position={[12, 5, 100]} color="orange" angle={1} penumbra={0.1} decay={0} intensity={2} castShadow={true} shadow-mapSize={512} />
        <pointLight position={[0, 0, 0]} decay={2} intensity={1} castShadow={true}/>
        <CombinedMesh/>
        <Html transform={false} fullscreen>
            <div className="flex h-screen w-screen items-center justify-center text-gray-950">
              <div className="flex flex-col w-[29vw] gap-7 items-center">
                <div className='flex items-center justify-center flex-col font-[200]'>
                  <h1 className='text-[50px]'>Sec<span className='font-[700]'>PRED</span></h1>
                  <h1 className='text-[17px]'>Created by <span className='font-[700]'>Filip Semrad</span>🤓</h1>
                </div>
                <h1 className='text-[23px] text-center'>
                  <span className='font-[700]'>Protein secondary structure</span> prediction and visualization tool created to help students🧑‍🎓 understand 🧬proteins🧬
                </h1>
                <div id="container">
                  <button className="learn-more">
                    <span className="circle" aria-hidden="true">
                      <span className="icon arrow"></span>
                    </span>
                    <a onClick={handleClick} className="button-text">Predict</a>
                  </button>
                </div>
              </div>
            </div>
        </Html>
        </Canvas>
      </>
  );
}