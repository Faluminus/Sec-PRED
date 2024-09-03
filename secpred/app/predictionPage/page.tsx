"use client"

import { Canvas } from '@react-three/fiber';
import { } from '@react-three/drei';
import {useState,useEffect,useRef} from 'react';
import { gql } from '@apollo/client';
import { ApolloClient, InMemoryCache } from '@apollo/client';

const PredictionPage = () =>{

    const [proteinID,setProteinID] = useState<string>()
    const [aminoAcidSeq,setAminoAcidSeq] = useState<string>()
    const [secStructureSeq,setSecStructureSeq] = useState<string>()
    const [numOfAminoAcids,setNumOfAminoAcids] = useState<number>()


    const [loading,setLoading] = useState<boolean>(false)

    const [proteinIDLoading,setProteinIDLoading] = useState<boolean>(false)
    
    const buttonRef = useRef<HTMLInputElement>(null);
    
    const client = new ApolloClient({
        uri: 'http://localhost:3000/api/graphql',
        cache: new InMemoryCache(),
    });
    
    useEffect(()=>{
        if(aminoAcidSeq != null){
            setNumOfAminoAcids(aminoAcidSeq?.length)
        }
        else{
            setNumOfAminoAcids(0)
        }
    },[aminoAcidSeq])

    const handlePrediction = async()=>{

        setLoading(true)

        await client
        .query({
            query: gql`
            query Query($aminoAcidSeq: String!) {
                secStructureSeq(aminoAcidSeq: $aminoAcidSeq)
            }
            `,
            variables: {
                aminoAcidSeq: aminoAcidSeq,  
            },
        })
        .then((result) => {
            setSecStructureSeq(result.data['secStructureSeq'])
            setLoading(false)
        });

    }
    
    const handleProteinID = async ()=>{
        if(!proteinIDLoading)
        {
            if(proteinID != '' && proteinID != null){
                setProteinIDLoading(true)
                if(buttonRef.current){
                    try{
                        buttonRef.current.classList.remove('bg-red-100');
                    }catch(exception){
                        console.log(exception)
                    }
                }
                await client
                .query({
                    query: gql`
                    query Query($aminoAcidSeqId: String!) {
                        aminoAcidSeq(id: $aminoAcidSeqId)
                    }
                    `,
                    variables: {
                        aminoAcidSeqId: proteinID,  
                    },
                })
                .then((result) => {
                    setAminoAcidSeq(result.data['aminoAcidSeq'])
                });
                setProteinIDLoading(false)
            }
            else{
                if(buttonRef.current){
                    buttonRef.current.classList.add('bg-red-100');
                }
            }
        }
    }
    return(<>
            <div className="w-screen h-screen p-10 pb-[55px] flex flex-row bg-gray-100 gap-4">
                <div className="flex flex-col h-[100%] w-[50vw] gap-3">
                    <div className='w-[50px] h-[50px]'>
                        <div className='absolute flex flex-row items-center gap-2'>
                            <div className='flex bg-blue-400 rounded-full w-[45px] h-[45px] justify-center items-center cursor-pointer shadow-2xl transition duration-200 hover:scale-110 hover:shadow-black' onClick={handleProteinID}>
                                <svg width="25" height="25" viewBox="0 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M11 20C15.9706 20 20 15.9706 20 11C20 6.02944 15.9706 2 11 2C6.02944 2 2 6.02944 2 11C2 15.9706 6.02944 20 11 20Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M18.9299 20.6898C19.4599 22.2898 20.6699 22.4498 21.5999 21.0498C22.4499 19.7698 21.8899 18.7198 20.3499 18.7198C19.2099 18.7098 18.5699 19.5998 18.9299 20.6898Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </div>
                            <input ref={buttonRef} value={proteinID} onChange={(e)=>{setProteinID(e.target.value)}} className=' bg-white w-[200px] h-[40px] flex items-center p-3 rounded-xl shadow-2xl font-[200]' placeholder='Protein id'></input>
                            {proteinIDLoading ? 
                            <span className="loading loading-spinner text-info w-[30px]"></span>
                            :<></>
                            }
                        </div>
                    </div>
                    
                    <div className="shadow-2xl rounded-2xl w-[100%] h-[20%] bg-white p-4">
                        <textarea value={aminoAcidSeq} onChange={(e)=>{setAminoAcidSeq(e.target.value)}} className="w-full h-full p-2 border rounded-md font-[200] resize-none" rows={4} cols={50} placeholder="Amino acid seq...">
                        </textarea>
                        <div className='flex w-full justify-end'>
                            <button  onClick={handlePrediction} className='w-[400px] h-[30px] bg-blue-400 rounded-2xl text-white shadow-2xl transition duration-200 hover:shadow-black hover:scale-x-[1.03]'>
                                Predict
                            </button>
                        </div>
                    </div>
                    <div className='flex flex-col'>
                        <p>Total amino acids: <span className='font-[700]'>{numOfAminoAcids}</span></p>
                        <p>Maximal handled amout: <span className='font-[700] text-red-500'>200</span></p>
                    </div>
                    <div className="shadow-2xl rounded-2xl flex flex-col bg-white p-5 mt-10 h-[30vh]">
                        <p>Sec<span className='font-[700]'>PRED</span><span className='font-[200]'>-String</span></p>
                        {loading ? 
                        <div className='flex items-center justify-center w-full h-full'>
                        <span className="loading loading-spinner text-info w-[3vw]"></span>
                        </div>
                        :
                        <p className='my-[2vh] mx-[1vw] text-blue-500'>{secStructureSeq}</p>
                        }
                    </div>
                    <div className="shadow-2xl rounded-2xl flex flex-col bg-white p-5 mt-1 h-[30vh]">
                        <p>Sec<span className='font-[700]'>PRED</span><span className='font-[200]'>-2D</span></p>
                        {loading ? 
                        <div className='flex items-center justify-center w-full h-full'>
                        <span className="loading loading-spinner text-info w-[3vw]"></span>
                        </div>
                        :
                        <Canvas>
                            <spotLight/>
                        </Canvas>
                        }
                    </div>
                    
                </div>
                
                <div className="bg-white shadow-2xl rounded-2xl w-[65vw] h-full p-5">
                    <p>SWISS-MODEL</p>
                    {loading ? 
                    <div className='flex items-center justify-center w-full h-full'>
                       <span className="loading loading-spinner text-info w-[5vw]"></span>
                    </div>
                    :
                    <Canvas>
                        <spotLight/>
                    </Canvas>
                    }
                </div>
            </div>
    </>)
}
export default PredictionPage