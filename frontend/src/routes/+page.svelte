<script>
    import { goto } from '$app/navigation';
    let input_ac = $state("");
    let input_search = $state("");

    async function searchPDB(){
        const response = await fetch('/fetchPDB',{
            method: "POST",
            body: JSON.stringify({
                'input_search': input_search
            })
        })
        let data = await response.json()
        input_ac = data.FASTA.FASTA
    }

    function PredictProtein(){
        fetch("http://127.0.0.1:5000/api/do-prediction",{
            method: "POST",
            body: JSON.stringify({
                AC: input_ac
            })
        })
        .then(response => response.json())
        .then(json => {
            goto(`/${json.ID}`)
        })
    }

</script>

<div class="flex flex-col items-center justify-center w-full h-full text-4xl gap-10 p-10">
    <h1 class="font-bold">SecPRED</h1>
    <div>
        <div class="flex flex-row gap-4">
            <div class="mb-5 w-[77px]">
                <input bind:value={input_search} oninput={() => input_search = input_search.toUpperCase()} type="text" id="id-input" maxlength="4" placeholder="4HB2" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
            </div>
            <button class="flex cursor-pointer items-center justify-center rounded-full bg-blue-300 w-[55px] h-[55px]" onclick={async () => await searchPDB()}>
                <svg width="25" height="25" viewBox="0 0 11 11" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5.0415 9.16669C7.31968 9.16669 9.1665 7.31986 9.1665 5.04169C9.1665 2.76351 7.31968 0.916687 5.0415 0.916687C2.76333 0.916687 0.916504 2.76351 0.916504 5.04169C0.916504 7.31986 2.76333 9.16669 5.0415 9.16669Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M8.67636 9.48285C8.91927 10.2162 9.47386 10.2895 9.90011 9.64785C10.2897 9.06118 10.033 8.57993 9.32719 8.57993C8.80469 8.57535 8.51136 8.98327 8.67636 9.48285Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        </div>
        <div class="flex flex-row gap-4 justify-center">
            <div class="mb-5 w-[60vw]">
                <input bind:value={input_ac} oninput={() => input_ac = input_ac.toUpperCase()} type="text" id="ac-input" placeholder="Amino acid sequence" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
            </div>
            <button class="flex cursor-pointer items-center justify-center rounded-full bg-blue-300 w-[55px] h-[55px]" onclick={PredictProtein}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M8.91016 19.92L15.4302 13.4C16.2002 12.63 16.2002 11.37 15.4302 10.6L8.91016 4.08002" stroke="white" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
                </svg> 
            </button>
        </div> 
    </div>
    <div class="flex flex-row items-center justify-center w-[80vw] h-full text-xl p-4 gap-24">
        <div class="gap-4 rounded-lg shadow-2xl bg-black bg-opacity-30 p-4">
            <div class="gap-3">
                <h4 class="font-bold">Prediction</h4>
                <p>
                    Bare string prediction from ml model.<br/>
                    Model uses <span class="cursor-pointer font-bold" data-tooltip-target="tooltip-right" data-tooltip-placement="right">dssp8</span> classification which means <br/> that model predicts 8 possible strutures: <br/><br/>
                    <span class="font-bold">Alpha helix (H)</span> <br/>
                    <span class="font-bold">Isolated Beta Bridge (B)</span>  <br/>
                    <span class="font-bold">Extended Strand in Beta Sheet (E)</span>  <br/>
                    <span class="font-bold">310 Helix (G)</span>  <br/>
                    <span class="font-bold">Pi Helix (I)</span> <br/>
                    <span class="font-bold">Turn (T)</span> <br/>
                    <span class="font-bold">Bend (S)</span> <br/>
                    <span class="font-bold">Coil or Random Coil (C)</span><br/><br/> 
                </p>
                <div class="bg-white w-[400px] rounded-lg p-2 my-2">
                    <p class='text-black'>HHHHHHHHH  EEEEEE TTS EEEETTEEEESSS HHHHHHHHHHHHTS  TTB</p>
                </div>
            </div>
        </div>
    </div>
</div>

