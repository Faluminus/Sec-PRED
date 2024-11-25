import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <div className='relative w-screen h-screen overflow-hidden'>
        <App />
      </div>
    </StrictMode>,
)
