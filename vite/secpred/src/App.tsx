

function App() {


  return (
    <>
    <img className="absolute w-[55rem] top-[-30vh] left-[25vw]" src="src\assets\images\green2.png"></img>
    <img className="absolute w-[55rem] top-[-60vh] left-[-10vw]" src="src\assets\images\proteinWhite.png"></img>
    <img className="absolute w-[40rem] top-[-30vh] right-[-15vw]" src="src\assets\images\blue.png"></img>
    <img className="absolute w-[40rem] top-[37vh] left-[-20vw]" src="src\assets\images\red.png"></img>
    <img className="absolute w-[40rem] top-[40vh] left-[10vw]" src="src\assets\images\green.png"></img>
    <img className="absolute w-[40rem] top-[40vh] left-[38vw]" src="src\assets\images\purple.png"></img>
    <img className="absolute w-[40rem] top-[45vh] right-[-5vw]" src="src\assets\images\yellow.png"></img>

    <div className='flex w-full h-full items-center justify-center'>
      <div className="flex flex-row items-center justify-center gap-[2vw] font-ApercuLight">
        <div className="flex flex-col w-[29rem] items-center justify-center gap-6 text-white">
          <div className="flex flex-col items-center justify-center gap-1">
            <h1 className="text-5xl">Sec<span className="font-ApercuMedium">PRED</span></h1>
            <h1>Created by <span className="font-ApercuMedium">Filip Semrad</span>🧑‍💻</h1>
          </div>
          <h1 className="text-center text-xl">
            Protein secondary structure prediction and visualization tool created to help students 🧑‍🎓 understand  🧬 proteins 🧬
          </h1>
        </div>
        <div className="w-[50vw] h-[90vh] bg-black rounded-lg opacity-55 backdrop-blur-lg shadow-2xl shadow-black">
          
        </div>
      </div>
    </div>
    </>
  )
}

export default App
