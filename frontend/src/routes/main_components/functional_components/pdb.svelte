<script>

    import { createEventDispatcher } from "svelte";
    let input_search = $state("");
    let {input_ac} = $props()
    const dispatch = createEventDispatcher(); 

    async function searchPDB(){
        const response = await fetch('/fetchPDB',{
            method: "POST",
            body: JSON.stringify({
                'input_search': input_search
            })
        })
        let data = await response.json()
        input_ac = data.FASTA.FASTA
        dispatch("updated", input_ac);
    }

    function handleKeyPressSearch(e){
        if(e.key == 'Enter'){
            searchPDB()
        }
    }
    
</script>


<div class="flex flex-row gap-4">
    <div class="mb-5 w-[77px]">
        <input bind:value={input_search} onkeydown={(e) => handleKeyPressSearch(e)} oninput={() => input_search = input_search.toUpperCase()} type="text" id="id-input" maxlength="4" placeholder="4HHB" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
    </div>
</div>
            
