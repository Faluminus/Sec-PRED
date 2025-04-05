<script>

    import { createEventDispatcher } from "svelte";
    import Cookies from "js-cookie"
    let input_search = $state("");
    let {input_ac} = $props()
    const dispatch = createEventDispatcher(); 
    const PUBLIC_API = import.meta.env.VITE_PUBLIC_API;
    async function searchPDB(){
        const response = await fetch(PUBLIC_API + `fetch-pdb/${input_search}`, { 
            method: "GET"
        })
        console.log(response)
        let data = await response.json()
        input_ac = data.output
        Cookies.set("input_ac", input_ac, {expires: 1})
        Cookies.set("pdb_code", input_search, {expires: 1} )
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
            
