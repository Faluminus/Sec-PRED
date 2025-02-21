<script>
    import { page } from '$app/stores';
	import { onMount } from 'svelte';

    let predID = $state($page.params.prediction);
    let path = $state("")
    let predValue = $state()

    async function fetchData(id) {
        let val = await fetch(`http://127.0.0.1:5000/api/get-by-id/${id}`)
            .then(response => response.json());
        return val;
    }
    
    function checkExistence(){
        if(predValue == null || predValue == undefined){
            return false
        }
        return true
    }

    function checkPending(){
        if (predValue.PENDING == false){
            return true
        }
        return false
    }

    function errorExistence(){
        if (predValue != null || predValue != undefined){
            if (predValue.ERROR == true){
                return true
            }
        }
        return false
    }

    function sleep(s) {
        return new Promise(resolve => setTimeout(resolve, s*1000));
    }

    onMount(async () => {
        let sleepTime = 2
        let noData = true
        while(noData){
            let val = await fetchData(predID);
            console.log(val)
            if (val != null || val != undefined) {
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
                    noData = false
                }     
            }
            await sleep(sleepTime)
            if (sleepTime < 32){
                sleepTime*=2
            }
        }
    })

</script>

<div class="w-screen h-screen p-10 pb-[55px] flex flex-row gap-4">
    {#if checkExistence()}
    {#if checkPending() && !errorExistence()}
    <div class="flex flex-col h-[100%] w-[50vw] gap-3">
        <div class='w-[50px] h-[50px]'>
            <div class='absolute flex flex-row items-center gap-5'>
                <div class='flex bg-blue-400 rounded-full w-[45px] h-[45px] justify-center items-center cursor-pointer shadow-2xl transition duration-200 hover:scale-110 hover:shadow-black'>
                    <svg width="25" height="25" viewBox="0 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M11 20C15.9706 20 20 15.9706 20 11C20 6.02944 15.9706 2 11 2C6.02944 2 2 6.02944 2 11C2 15.9706 6.02944 20 11 20Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M18.9299 20.6898C19.4599 22.2898 20.6699 22.4498 21.5999 21.0498C22.4499 19.7698 21.8899 18.7198 20.3499 18.7198C19.2099 18.7098 18.5699 19.5998 18.9299 20.6898Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class='flex bg-blue-400 rounded-full w-[45px] h-[45px] justify-center items-center cursor-pointer shadow-2xl transition duration-200 hover:scale-110 hover:shadow-black'>
                    <svg width="25" height="25" viewBox="0 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M11 20C15.9706 20 20 15.9706 20 11C20 6.02944 15.9706 2 11 2C6.02944 2 2 6.02944 2 11C2 15.9706 6.02944 20 11 20Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M18.9299 20.6898C19.4599 22.2898 20.6699 22.4498 21.5999 21.0498C22.4499 19.7698 21.8899 18.7198 20.3499 18.7198C19.2099 18.7098 18.5699 19.5998 18.9299 20.6898Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class='flex bg-blue-400 rounded-full w-[45px] h-[45px] justify-center items-center cursor-pointer shadow-2xl transition duration-200 hover:scale-110 hover:shadow-black'>
                    <svg width="25" height="25" viewBox="0 0 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M11 20C15.9706 20 20 15.9706 20 11C20 6.02944 15.9706 2 11 2C6.02944 2 2 6.02944 2 11C2 15.9706 6.02944 20 11 20Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M18.9299 20.6898C19.4599 22.2898 20.6699 22.4498 21.5999 21.0498C22.4499 19.7698 21.8899 18.7198 20.3499 18.7198C19.2099 18.7098 18.5699 19.5998 18.9299 20.6898Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <span class="loading loading-spinner text-info w-[30px]"></span>
            </div>
        </div>
        <div class="shadow-2xl rounded-2xl flex flex-col bg-white p-5 mt-10 h-[30vh] text-black overflow-scroll">
            <p class="fixed">Sec<span class='font-[700]'>PRED</span><span class='font-[200]'>-CONV</span></p>
            <div class='flex items-center justify-center w-full h-full'>
            <span class="loading loading-spinner text-info w-[3vw]"></span>
            </div>
            <p class='my-[2vh] mx-[1vw] text-blue-500'>
                {predValue.SSCONV}
            </p>
        </div>
        <div class="shadow-2xl rounded-2xl flex flex-col bg-white p-5 mt-1 h-[30vh] text-black overflow-scroll">
            <p class="fixed">Sec<span class='font-[700]'>PRED</span><span class='font-[200]'>-LSTM</span></p>
            <div class='flex items-center justify-center w-full h-full'>
            <span class="loading loading-spinner text-info w-[3vw]"></span>
            </div>
            <p class='my-[2vh] mx-[1vw] text-blue-500'>
                {predValue.SSLSTM}
            </p>
        </div>
        <div class="shadow-2xl rounded-2xl flex flex-col bg-white p-5 mt-1 h-[30vh] text-black">
            <p>Sec<span class='font-[700]'>PRED</span><span class='font-[200]'>-2D</span></p>
            <div class='flex items-center justify-center w-full h-full'>
            <span class="loading loading-spinner text-info w-[3vw]"></span>
            </div>
            <svg class="border-gray-300 rounded-lg h-80" height={predValue.XYHEIGTH} viewBox={`256 0 1000 200`}  xmlns="http://www.w3.org/2000/svg">
                <path d={path} stroke="blue" fill="none" stroke-width="2"/>
            </svg>
        </div>
    </div>
    <div class="shadow-2xl rounded-2xl flex flex-col bg-black p-5 mt-1 text-white bg-opacity-30 w-[50vw] h-full">
        <p>Sec<span class='font-[700]'>PRED</span><span class='font-[200]'>-2D</span></p>
        <div class='flex items-center justify-center w-full h-full'>
        <span class="loading loading-spinner text-info w-[3vw]"></span>
        </div>
    </div>
    {/if}
    {#if !checkPending() && !errorExistence()}
    <div class="flex flex-row items-center justify-center w-full h-full">
        <h1 class="text-2xl">The prediction is running...</h1>
        <img src="/White Dog Running Sticker.gif">
    </div>
    {/if}
    {#if errorExistence()}
    <div class="flex flex-row items-center justify-center w-full h-full gap-5">
        <h1 class="text-2xl">Something went wrong</h1>
        <img width="300" src="/Dog Crying Sticker by Sticker Book iOS GIFs.gif">
    </div>
    {/if}
    {/if}
</div>