<script>

    import { goto } from '$app/navigation';
    export let input_ac;
    const PUBLIC_API = import.meta.env.VITE_PUBLIC_API;
    function PredictProtein(){
        if(input_ac !== ""){
            fetch(PUBLIC_API + "do-prediction",{
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
    }

    function handleKeyPressPredict(e){
        if(e.key == 'Enter'){
            PredictProtein()
        }
    }
</script>
<div class="flex flex-row gap-4 justify-center">
    <div class="mb-5 w-[60vw]">
        <input bind:value={input_ac} onkeydown={(e) => handleKeyPressPredict(e)} oninput={() => input_ac = input_ac.toUpperCase()} type="text" id="ac-input" placeholder="Amino acid sequence" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
    </div>
    <button aria-label="Submit" class="flex cursor-pointer items-center justify-center rounded-full bg-blue-500 w-[55px] h-[55px]" onclick={PredictProtein}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M8.91016 19.92L15.4302 13.4C16.2002 12.63 16.2002 11.37 15.4302 10.6L8.91016 4.08002" stroke="white" stroke-width="1.5" stroke-miterlimit="10" stroke-linecap="round" stroke-linejoin="round"/>
        </svg> 
    </button>
</div> 


