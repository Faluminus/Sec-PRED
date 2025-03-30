<script>
	import Cookies from "js-cookie";
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import Visual from './prediction_components/functional_components/visual.svelte';
	import Datacard from './prediction_components/functional_components/datacard.svelte';
	import Goback from './prediction_components/functional_components/goback.svelte';
	import Printpdf from './prediction_components/functional_components/printpdf.svelte';
	import Printfasta from './prediction_components/functional_components/printfasta.svelte';
	import SelectModel from "./prediction_components/functional_components/SelectModel.svelte";
	import ProteinVisExplain from "./prediction_components/nonlogic_components/ProteinVisExplain.svelte";

	let predID;
	let path = $state('');
	let predValue = $state();
	let selected = $state('LSTM+CNN');
	let timeWaited = $state(0);
	let secondaryStructure = $state('');


	async function fetchData(id) {
		let val = await fetch(`http://localhost:5001/api/get-by-id/${id}`).then((response) =>
			response.json()
		);
		return val;
	}

	function checkExistence() {
		if (predValue == null || predValue == undefined) {
			return false;
		}
		return true;
	}

	function checkPending() {
		if (predValue.PENDING == false) {
			return true;
		}
		return false;
	}

	function errorExistence() {
		if (predValue != null || predValue != undefined) {
			if (predValue.ERROR == true) {
				return true;
			}
		}
		return false;
	}

	function sleep(s) {
		return new Promise((resolve) => setTimeout(resolve, s * 1000));
	}

	function handleModelChange(){
		if(selected == "LSTM+CNN"){
			secondaryStructure = predValue.SSLSTM
		} else if(selected == "CNN"){
			secondaryStructure = predValue.SSCONV
		} else {
			secondaryStructure = predValue.SSTRANSFORMER
		}
	}

	async function countTime(){
		let x = await sleep(1);
		timeWaited++;
	}

	//Fetching
	onMount(async () => {
		predID = $page.params.prediction
		let sleepTime = 2;
		let noData = true;
		while (noData) {
			let val = await fetchData(predID);
			if (val != null && val != undefined) {
				predValue = val;
				console.log(predValue)
				if (!predValue.PENDING || predValue.ERROR) {
					noData = false;
					console.log(predValue)
					predValue.AC = predValue.AC.slice(1,-1)
					secondaryStructure = predValue.SSLSTM
					Cookies.set("secondary_structure_lstm", predValue.SSLSTM, {expires: 1})
					Cookies.set("secondary_structure_conv", predValue.SSCONV, {expires: 1})
				}
			}
			for(let i = 0; i < sleepTime; i++){
				await countTime();
			}
			if (sleepTime < 32) {
				sleepTime *= 2;
			}
		}
	});

	
</script>

<div class="flex h-screen w-screen items-center justify-center flex-row gap-4 p-10 pb-[55px]">
	{#if checkExistence()}
		{#if checkPending() && !errorExistence()}
			<div class="flex items-center justify-center h-[100%] w-full flex-row gap-3">
				<div class="flex flex-row gap-5 justify-center items-center w-full h-full">
					<div class="flex flex-col gap-4 w-full h-full items-center">
						<div class="flex flex-row gap-14 w-[80vw] items-end fixed">
							<SelectModel bind:selected={selected} bind:handleModelChange={handleModelChange}></SelectModel>
							<div class="flex flex-row gap-4">
								<Printpdf></Printpdf>
								<Printfasta></Printfasta>
							</div>
						</div>
						<ProteinVisExplain></ProteinVisExplain>
						<div class="w-[80vw] flex-col rounded-2xl bg-gray-100 p-5 text-black shadow-md inline-flex my-28">
							<p>
							Sec<span class="font-[700]">PRED</span><span class="font-[200]">-{selected}</span>
							</p>
							<Visual bind:aminoAcid={predValue.AC} bind:secondaryStructure={secondaryStructure}/>
						</div>
					</div>
				</div>
				<div class="fixed bottom-10 left-10">
					<Goback></Goback>
				</div>
			</div>
		{/if}
		{#if !checkPending() && !errorExistence()}
			<div class="flex h-full w-full flex-col items-center justify-center">
				<div class="flex h-full w-full flex-row items-center justify-center">
					<h1 class="text-2xl">The prediction is running...</h1>
					<img alt="Running dog" src="/White Dog Running Sticker.gif" />
				</div>
				<div>
					<h3>Task pending {timeWaited} seconds</h3>
				</div>
			</div>
		{/if}
		{#if errorExistence()}
			<div class="flex h-full w-full flex-row items-center justify-center gap-5">
				<h1 class="text-2xl">Something went wrong</h1>
				<img alt="Crying dog" width="300" src="/Dog Crying Sticker by Sticker Book iOS GIFs.gif" />
			</div>
		{/if}
	{/if}
</div>
