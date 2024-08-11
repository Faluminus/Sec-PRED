import React from 'react'

export default function Footer_component(props:any){
    return (
        <footer className="absolute bottom-0 w-full h-10 px-5 py-9">
            <div className="flex flex-row w-full justify-between">
                <div className="font-[200] flex flex-row gap-4">
                    <h3>®Sec<span className='font-[700]'>PRED</span></h3>
                    <h3>Cookies</h3>
                </div>
                <div className="font-[200] flex flex-row gap-4">
                    <h3>www.secpred.com</h3>
                </div>
            </div>
        </footer>
  )
}

