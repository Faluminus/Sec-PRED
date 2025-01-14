'use client'

import {motion} from 'framer-motion'

export default function Transition({
    children
}:{
    children: React.ReactNode
}){
    return(
        <motion.div transition={{ ease: "easeOut", duration: 2 }}>
            {children}
        </motion.div>
    ) 
}