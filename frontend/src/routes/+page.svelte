<script>
	import Pdb from './main_components/functional_components/pdb.svelte';
	import Predict from './main_components/functional_components/predict.svelte';
    let input_ac = $state("");

    function handleUpdate(event){
        input_ac = event.detail;
        console.log(input_ac)
    }

</script>

<div class="flex flex-col items-left justify-center w-full h-full text-4xl gap-4 p-32 pl-64 pr-64">
    <h1 class="font-bold"><span class='font-bold'>Sec</span><span class='text-blue-500'>PRED</span></h1>
    <span class="w-full h-1 bg-black"></span>
    <h2 class='text-[15px] leading-snug'><span class='font-bold'>Sec</span><span class='text-blue-500 font-bold'>PRED</span> is a web‐based tool for protein secondary structure prediction
         that employs machine learning models, including <span class='font-bold'>RNN + CNN, CNN, and Transformer architectures.</span> Users 
         may provide input either as a <span class='font-bold'>PDB ID</span>—which is submited after pressing enter and retrieves the associated amino acid sequence—or 
         by entering a sequence manually. The tool generates predictions in <a class='cursor-pointer font-bold text-blue-500'>Q8 format</a> accompanied by <span class='font-bold'>visualizations in <a class='cursor-pointer text-blue-500' href="">Q3 format</a></span>, 
         and the results page includes a detailed explanation of the underlying methodology. Additionally, users have 
         the option to select which model outputs to display. For further details regarding the methodology and usage, 
         please consult the documentation. The service is freely accessible without requiring user registration.</h2>
    <div class='flex flex-col gap-5 text-2xl items-center justify-center my-16'>
        <h3>Insert PDB ID or amino acid sequence to predict</h3>
        <div class="flex flex-row gap-3">
            <Pdb on:updated={handleUpdate}></Pdb>
            <Predict bind:input_ac={input_ac}></Predict>
        </div>   
    </div>
</div>

