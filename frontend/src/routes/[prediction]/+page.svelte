<script>
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import Visual from './prediction_components/functional_components/visual.svelte';
	import Datacard from './prediction_components/functional_components/datacard.svelte';

	let predID;
	let path = $state('');
	let predValue = $state();
	let selected = $state('LSTM+CNN');
	let timeWaited = $state(0);
	let secondaryStructure = $state('');


	async function fetchData(id) {
		let val = await fetch(`http://192.168.17.167:5000/api/get-by-id/${id}`).then((response) =>
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

	//time loging
	onMount(async () => {
		while (checkPending()) {
			let x = await sleep(1);
			timeWaited++;
		}
	});

	//Fetching
	onMount(async () => {
		predID = $page.params.prediction
		let sleepTime = 2;
		let noData = true;
		while (noData) {
			let val = await fetchData(predID);
			console.log(val);
			if (val != null && val != undefined) {
				predValue = val;
				if (!predValue.PENDING || predValue.ERROR) {
					let arr = JSON.parse(predValue.XY);
					arr.forEach((item, index) => {
						if (index == 0) {
							path += `M ${item[0]} ${item[1]}`;
						} else {
							path += `L ${item[0]} ${item[1]}`;
						}
					});
					noData = false;
					predValue.AC = predValue.AC.slice(1,-1)
					secondaryStructure = predValue.SSLSTM
				}
			}
			await sleep(sleepTime);
			if (sleepTime < 32) {
				sleepTime *= 2;
			}
		}
	});

	
</script>

<div class="flex h-screen w-screen flex-row gap-4 p-10 pb-[55px]">
	{#if checkExistence()}
		{#if checkPending() && !errorExistence()}
			<div class="flex h-[100%] w-full flex-row gap-3">
				<div class="flex justify-center h-full pt-12">
					<Datacard time={timeWaited}></Datacard>
				</div>
				<div class="flex flex-col h-full w-full items-center">
					<div class="flex items-left h-[80px] w-full  justify-center">
						<form class="mx-auto h-[60px] w-[70vw]">
							<label for="models" class="mb-2 block text-sm font-medium text-gray-900"
								>Pick a model</label
							>
							<select
								bind:value={selected}
								on:change={handleModelChange}
								id="models"
								class="block w-full rounded-lg border border-gray-300 bg-gray-50 p-2.5 text-sm text-gray-900 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500"
							>
								<option value="LSTM+CNN">LSTM+CNN - 81%</option>
								<option value="CNN">CNN - 74%</option>
								<option value="TRANSFORMER">Transformer</option>
							</select>
						</form>
						<div
							class="flex h-[45px] w-[45px] cursor-pointer items-center justify-center rounded-full bg-blue-400 shadow-2xl transition duration-200 hover:scale-110 hover:shadow-black fixed right-10"
							on:click={print()}
						>
							<svg
								width="25"
								height="25"
								viewBox="0 0 25 25"
								fill="none"
								xmlns="http://www.w3.org/2000/svg"
								
							>
								<path
									d="M11 20C15.9706 20 20 15.9706 20 11C20 6.02944 15.9706 2 11 2C6.02944 2 2 6.02944 2 11C2 15.9706 6.02944 20 11 20Z"
									stroke="white"
									stroke-width="1.5"
									stroke-linecap="round"
									stroke-linejoin="round"
								/>
								<path
									d="M18.9299 20.6898C19.4599 22.2898 20.6699 22.4498 21.5999 21.0498C22.4499 19.7698 21.8899 18.7198 20.3499 18.7198C19.2099 18.7098 18.5699 19.5998 18.9299 20.6898Z"
									stroke="white"
									stroke-width="1.5"
									stroke-linecap="round"
									stroke-linejoin="round"
								/>
							</svg>
						</div>
					</div>
					<div class="flex flex-row items-center justify-center gap-2">
						<h3 class='font-bold'>Protein secondary structures: </h3> 
						<svg width="24" height="12"><g><defs><g id="icon-ss-Hb-56cf"><path d="M 4.8 0 H 1.6 Q 0 0, -4.8 12 H -1.6 Q 0 12, 4.8 0" fill="#ff6600"></path></g><g id="icon-ss-Hf-56cf"><path d="M -4.8 0 H -1.6 Q 0 0, 4.8 12 H 1.6 Q 0 12, -4.8 0" fill="#ff9900"></path></g><g id="icon-ss-Hef-56cf" style="fill: rgb(255, 153, 0);"><path d="M 0 4.800000000000001 L 6.4 12 H 3.2 Q 1.6 12, 0 7.199999999999999"></path></g><g id="icon-ss-Heb-56cf"><path d="M 8 4.800000000000001 L 1.6 12 H 4.8 Q 6.4 12, 8 7.199999999999999" fill="#ff6600"></path></g><clipPath id="clip56cf"><rect x="0" y="0" width="24" height="12"></rect></clipPath></defs><g clip-path="url(#clip56cf)"><rect width="0" height="2.4" y="4.8" x="0" fill="#cc3399"></rect><use xlink:href="#icon-ss-Hb-56cf" transform="translate(8, 0)"></use><use xlink:href="#icon-ss-Hef-56cf" transform="translate(0, 0)"></use><use xlink:href="#icon-ss-Heb-56cf" transform="translate(16, 0)"></use><use xlink:href="#icon-ss-Hf-56cf" transform="translate(16, 0)"></use></g></g></svg>
						<p>Helix</p>
						<svg width="16" height="12"><g><defs><g id="icon-ss-Hb-acc4"><path d="M 4.8 0 H 1.6 Q 0 0, -4.8 12 H -1.6 Q 0 12, 4.8 0" fill="#ff6600"></path></g><g id="icon-ss-Hf-acc4"><path d="M -4.8 0 H -1.6 Q 0 0, 4.8 12 H 1.6 Q 0 12, -4.8 0" fill="#ff9900"></path></g><g id="icon-ss-Hef-acc4" style="fill: rgb(255, 153, 0);"><path d="M 0 4.800000000000001 L 6.4 12 H 3.2 Q 1.6 12, 0 7.199999999999999"></path></g><g id="icon-ss-Heb-acc4"><path d="M 8 4.800000000000001 L 1.6 12 H 4.8 Q 6.4 12, 8 7.199999999999999" fill="#ff6600"></path></g><clipPath id="clipacc4"><rect x="0" y="0" width="16" height="12"></rect></clipPath></defs><g clip-path="url(#clipacc4)"><rect width="0" height="2.4" y="4.8" x="0" fill="#cc3399"></rect><rect width="4" height="2.4" y="4.8" x="12" fill="#cc3399"></rect><path d="M 0 2.4000000000000004 H 11.6 V 0 L 16 6 L 11.6 12 V 9.600000000000001 H 0 " fill="#660099"></path></g></g></svg>
						<p>Strand</p>
						<svg width="16" height="12"><g><defs><g id="icon-ss-Hb-4494"><path d="M 4.8 0 H 1.6 Q 0 0, -4.8 12 H -1.6 Q 0 12, 4.8 0" fill="#ff6600"></path></g><g id="icon-ss-Hf-4494"><path d="M -4.8 0 H -1.6 Q 0 0, 4.8 12 H 1.6 Q 0 12, -4.8 0" fill="#ff9900"></path></g><g id="icon-ss-Hef-4494" style="fill: rgb(255, 153, 0);"><path d="M 0 4.800000000000001 L 6.4 12 H 3.2 Q 1.6 12, 0 7.199999999999999"></path></g><g id="icon-ss-Heb-4494"><path d="M 8 4.800000000000001 L 1.6 12 H 4.8 Q 6.4 12, 8 7.199999999999999" fill="#ff6600"></path></g><clipPath id="clip4494"><rect x="0" y="0" width="16" height="12"></rect></clipPath></defs><g clip-path="url(#clip4494)"><rect width="16" height="2.4" y="4.8" x="0" fill="#cc3399"></rect></g></g></svg>
						<p>Coil</p>
					</div>
					<div
						class="mt-4 flex w-[70vw] flex-col rounded-2xl bg-gray-100 p-5 text-black shadow-2xl"
					>
						<p>
							Sec<span class="font-[700]">PRED</span><span class="font-[200]">-{selected}</span>
						</p>
						<Visual bind:aminoAcid={predValue.AC} bind:secondaryStructure={secondaryStructure}/>
					</div>
				</div>
			</div>
		{/if}
		{#if !checkPending() && !errorExistence()}
			<div class="flex h-full w-full flex-col items-center justify-center">
				<div class="flex h-full w-full flex-row items-center justify-center">
					<h1 class="text-2xl">The prediction is running...</h1>
					<img src="/White Dog Running Sticker.gif" />
				</div>
				<div>
					<h3>Task pending {timeWaited} seconds</h3>
				</div>
			</div>
		{/if}
		{#if errorExistence()}
			<div class="flex h-full w-full flex-row items-center justify-center gap-5">
				<h1 class="text-2xl">Something went wrong</h1>
				<img width="300" src="/Dog Crying Sticker by Sticker Book iOS GIFs.gif" />
			</div>
		{/if}
	{/if}
</div>
