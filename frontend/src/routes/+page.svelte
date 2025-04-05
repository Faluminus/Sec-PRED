<script>
	import Pdb from './main_components/functional_components/pdb.svelte';
	import Predict from './main_components/functional_components/predict.svelte';
	import Visprotein from './main_components/functional_components/visprotein.svelte';
    let input_ac = $state("");

    function handleUpdate(event){
        input_ac = event.detail;
        console.log(input_ac)
    }

    const hemoglobinAlphaOne = "Hemoglobin is a protein that facilitates the transport of oxygen and other gases in red blood cells. Almost all vertebrates contain hemoglobin. It consists of four protein subunits (globins), and is one of the first proteins whose 3D structure has been experimentally determined. There are many types of hemoglobin, with hemoglobin alpha 1 (encoded by the HBA1 gene) occurring in humans and being the main form of hemoglobin in adults. In this use case, AlphaFind shows us that highly similar hemoglobin structures can also be found in other species."
    const cytochromeP450 = "Cytochrome P450 are enzymes which are important for the metabolism of many endogenous compounds and xenobiotics. P450 enzymes have been identified across all biological kingdoms: animals, plants, fungi, bacteria and archaea, as well as in viruses. Cytochrome P450 proteins contain one chain which is composed of more than 20 sheets and helices. Their sequence similarity is very low. In this use case, we can see similarities among cytochrome P450 structures from various species. The search starts with a cytochrome from corn (Zea Mays), and within the first 50 hits, we find similar structures originating from various animals (mouse, cat, horse, etc.)"
    const pin5Protein = "The PIN proteins are transmembrane proteins which regulate plant growth by influencing auxin transport from the cytosol to the extracellular space. They only occur in plants and feature a configuration of 10 main helices that collectively form a pore. Eight types of PIN proteins are known (PIN1-PIN8), and recently, the structures of three PIN proteins were uncovered and published in Nature. The structure of the PIN5 protein differs from other PINs (Ung 2022) and has not yet been experimentally determined. This use case shows that the PIN5 protein structure is strongly conserved among many different plant species."
</script>

<div class="flex flex-col items-left justify-center w-full h-full text-4xl gap-4 pt-20 pl-[10vw] pr-[10vw]">
    <h1 class="font-bold"><span class='font-bold'>Sec</span><span class='text-blue-500'>PRED</span></h1>
    <span class="w-full h-1 bg-black"></span>
    <h2 class='text-[15px] leading-snug'><span class='font-bold'>Sec</span><span class='text-blue-500 font-bold'>PRED</span> is a web‐based tool for protein secondary structure prediction
         that employs machine learning models, including <span class='font-bold'>RNN + CNN, CNN, and Transformer architectures.</span> Users 
         may provide input either as a <span class='font-bold'>PDB ID</span>—which is submited after pressing enter and retrieves the associated amino acid sequence—or 
         by entering a sequence manually. The tool generates predictions in <span class='cursor-pointer font-bold text-blue-500'>Q8 format</span> accompanied by <span class='font-bold'>visualizations in <a class='cursor-pointer text-blue-500' href="">Q3 format</a></span>, 
         and the results page includes a detailed explanation of the underlying methodology. Additionally, users have 
         the option to select which model outputs to display. For further details regarding the methodology and usage, 
         please consult the documentation. The service is freely accessible without requiring user registration.</h2>
    <div class='flex flex-col gap-5 text-2xl items-center justify-center my-16 w-full'>
        <h3>Insert PDB ID or amino acid sequence to predict</h3>
        <div class="flex flex-row gap-3">
            <Pdb on:updated={handleUpdate}></Pdb>
            <Predict bind:input_ac={input_ac}></Predict>
        </div>
    </div>
    <div class="flex flex-row items-center justify-center w-full gap-7">
        <Visprotein protein_code="4HHB" protein_name="Hemoglobin alpha 1" protein_description={hemoglobinAlphaOne}></Visprotein>
        <Visprotein protein_code="4HHB" protein_name="Cytochrome P450" protein_description={cytochromeP450}></Visprotein>
        <Visprotein protein_code="4HHB" protein_name="Pin5 protein" protein_description={pin5Protein}></Visprotein>
    </div>
</div>

