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

    function handleKeyPressSearch(e){
        if(e.key == 'Enter'){
            searchPDB()
        }
    }

    function handleKeyPressPredict(e){
        if(e.key == 'Enter'){
            PredictProtein()
        }
    }

</script>

<div class="flex flex-col items-left justify-center w-full h-full text-4xl gap-4 p-32 pl-64 pr-64">
    <h1 class="font-bold"><span class='font-bold'>Sec</span><span class='text-blue-500'>PRED</span></h1>
    <span class="w-full h-1 bg-black"></span>
    <h2 class='text-[15px] leading-snug'><span class='font-bold'>Sec</span><span class='text-blue-500 font-bold'>PRED</span> is a web‐based tool for protein secondary structure prediction
         that employs machine learning models, including <span class='font-bold'>RNN + CNN, CNN, and Transformer architectures.</span> Users 
         may provide input either as a <span class='font-bold'>PDB ID</span>—which is automatically submited after 2sec of inaction and retrieves the associated amino acid sequence—or 
         by entering a sequence manually. The tool generates predictions in <a class='cursor-pointer font-bold text-blue-500'>Q8 format</a> accompanied by <span class='font-bold'>visualizations in <a class='cursor-pointer text-blue-500' href="">Q3 format</a></span>, 
         and the results page includes a detailed explanation of the underlying methodology. Additionally, users have 
         the option to select which model outputs to display. For further details regarding the methodology and usage, 
         please consult the documentation. The service is freely accessible without requiring user registration.</h2>
    <div class='flex flex-col gap-5 text-2xl items-center justify-center my-16'>
        <h3>Insert PDB ID or amino acid sequence to predict</h3>
        <div class="flex flex-row gap-3">
            <div class="flex flex-row gap-4">
                <div class="mb-5 w-[77px]">
                    <input bind:value={input_search} onkeydown={(e) => handleKeyPressSearch(e)} oninput={() => input_search = input_search.toUpperCase()} type="text" id="id-input" maxlength="4" placeholder="4HB2" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
                </div>
            </div>
            <div class="flex flex-row gap-4 justify-center">
                <div class="mb-5 w-[60vw]">
                    <input bind:value={input_ac} onkeydown={(e) => handleKeyPressPredict(e)} oninput={() => input_ac = input_ac.toUpperCase()} type="text" id="ac-input" placeholder="Amino acid sequence" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
                </div>
                <button class="flex cursor-pointer items-center justify-center rounded-full bg-blue-500 w-[55px] h-[55px]" onclick={PredictProtein}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8.91016 19.92L15.4302 13.4C16.2002 12.63 16.2002 11.37 15.4302 10.6L8.91016 4.08002" stroke="white" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg> 
                </button>
            </div> 
        </div>   
    </div>
</div>

