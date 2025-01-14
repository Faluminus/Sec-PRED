<script>
    let input_ac = $state("");
    let input_search = $state("");

    function SearchPDB(){

    }

    function PredictProtein(){
        fetch("127.0.0.1:5000/api/do-prediction")
        .then()
    }

    function ToUpper(){

    }

</script>

<div class="flex flex-col items-center justify-center w-full h-full text-4xl gap-10 p-10">
    <h1 class="font-bold">SecPRED</h1>
    <div>
        <div class="flex flex-row gap-4">
            <div class="mb-5 w-[77px]">
                <input bind:value={input_search} type="text" id="id-input" maxlength="4" placeholder="4HB2" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
            </div>
            <button class="flex cursor-pointer items-center justify-center rounded-full bg-blue-300 w-[55px] h-[55px]" onclick={SearchPDB}>
                <svg width="25" height="25" viewBox="0 0 11 11" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M5.0415 9.16669C7.31968 9.16669 9.1665 7.31986 9.1665 5.04169C9.1665 2.76351 7.31968 0.916687 5.0415 0.916687C2.76333 0.916687 0.916504 2.76351 0.916504 5.04169C0.916504 7.31986 2.76333 9.16669 5.0415 9.16669Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M8.67636 9.48285C8.91927 10.2162 9.47386 10.2895 9.90011 9.64785C10.2897 9.06118 10.033 8.57993 9.32719 8.57993C8.80469 8.57535 8.51136 8.98327 8.67636 9.48285Z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </button>
        </div>
        <div class="flex flex-row gap-4 justify-center">
            <div class="mb-5 w-[60vw]">
                <input bind:value={input_ac} type="text" id="ac-input" placeholder="Amino acid sequence" class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-white text-base">
            </div>
            <button class="flex cursor-pointer items-center justify-center rounded-full bg-blue-300 w-[55px] h-[55px]" onclick={SearchPDB}>
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
        <div class="gap-4 rounded-lg shadow-2xl bg-black bg-opacity-30 p-4">
            <div class="gap-3">
                <h4 class="font-bold">Visualisation</h4>
                <p>
                    Visualisation of predicted dssp structures
                </p>
                <div class="bg-white w-[350px] rounded-lg p-2 my-2">
                    <svg width="322" height="66" viewBox="0 0 322 66" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M1 3C2.76667 3.41491 4.53333 4.23615 6.3 5.46374C8.06667 7.45327 9.83333 9.78269 11.6 12.452C13.3667 15.6907 15.1333 19.1473 16.9 22.8216C18.6667 26.7798 20.4333 30.7982 22.2 34.8766C23.9667 38.9068 25.7333 42.8296 27.5 46.6451C29.2667 50.0881 31.0333 53.2738 32.8 56.2021C34.5667 58.4947 36.3333 60.4222 38.1 61.9844C39.8667 62.7517 41.6333 63.1057 43.4 63.0463C45.1667 62.1627 46.9333 60.8853 48.7 59.214C50.4667 56.8241 52.2333 54.1242 54 51.1144C55.7667 47.6091 57.5333 43.9283 59.3 40.0723C61.0667 36.0249 62.8333 31.9654 64.6 27.8938C66.3667 23.9664 68.1333 20.1921 69.9 16.571C71.6667 13.406 73.4333 10.5342 75.2 7.95593C76.9667 6.07091 78.7333 4.57149 80.5 3.45766C82.2667 3.16105 84.0333 3.27917 85.8 3.81202C87.5667 5.15232 89.3333 6.86867 91.1 8.96104C92.8667 11.719 94.6333 14.7528 96.4 18.0625C98.1667 21.7871 99.9333 25.6421 101.7 29.6277C103.467 33.7096 105.233 37.7554 107 41.7649C108.767 45.5365 110.533 49.1112 112.3 52.489C114.067 55.3333 115.833 57.8522 117.6 60.0457C119.367 61.4976 121.133 62.5487 122.9 63.199C124.667 63.021 126.433 62.4324 128.2 61.4332C129.967 59.6543 131.733 57.5223 133.5 55.0371C135.267 51.9484 137.033 48.6216 138.8 45.0569C140.567 41.1635 142.333 37.1862 144.1 33.125C145.867 29.0638 147.633 25.0865 149.4 21.1931C151.167 17.6284 152.933 14.3016 154.7 11.2129C156.467 8.72769 158.233 6.59566 160 4.81676C161.767 3.81763 163.533 3.22904 165.3 3.05097C167.067 3.70131 168.833 4.75243 170.6 6.20431C172.367 8.39776 174.133 10.9167 175.9 13.761C177.667 17.1388 179.433 20.7135 181.2 24.485C182.967 28.4946 184.733 32.5404 186.5 36.6223C188.267 40.6079 190.033 44.4629 191.8 48.1875C193.567 51.4971 195.333 54.531 197.1 57.289C198.867 59.3813 200.633 61.0977 202.4 62.438C204.167 62.9708 205.933 63.0889 207.7 62.7923C209.467 61.6785 211.233 60.1791 213 58.2941C214.767 55.7158 216.533 52.844 218.3 49.679C220.067 46.0579 221.833 42.2836 223.6 38.3562C225.367 34.2846 227.133 30.2251 228.9 26.1777C230.667 22.3216 232.433 18.6409 234.2 15.1356C235.967 12.1258 237.733 9.42591 239.5 7.03598C241.267 5.3647 243.033 4.08727 244.8 3.20369C246.567 3.14432 248.333 3.49828 250.1 4.26556C251.867 5.82781 253.633 7.75526 255.4 10.0479C257.167 12.9762 258.933 16.1619 260.7 19.6049C262.467 23.4204 264.233 27.3432 266 31.3734C267.767 35.4518 269.533 39.4702 271.3 43.4284C273.067 47.1027 274.833 50.5593 276.6 53.798C278.367 56.4673 280.133 58.7967 281.9 60.7863C283.667 62.0138 285.433 62.8351 287.2 63.25C288.967 62.8351 290.733 62.0138 292.5 60.7863C294.267 58.7967 296.033 56.4673 297.8 53.798C299.567 50.5593 301.333 47.1027 303.1 43.4284C304.867 39.4702 306.633 35.4518 308.4 31.3734C310.167 27.3432 311.933 23.4204 313.7 19.6049C315.467 16.1619 317.233 12.9762 319 10.0479" stroke="black" stroke-width="5"/>
                    </svg> 
                </div>
            </div>
        </div>
    </div>
</div>

