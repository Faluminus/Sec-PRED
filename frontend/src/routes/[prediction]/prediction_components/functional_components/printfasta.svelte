<script>
	import { text } from '@sveltejs/kit';
    import Cookies from 'js-cookie';
    function handleFastaDownload(){
        const code = Cookies.get("pdb_code")
        const ssconv = ">ss_convolutional nn\n" + Cookies.get("secondary_structure_conv");
        const sslstm = ">ss_recurent_nn\n" + Cookies.get("secondary_structure_lstm");
        const input_ac = ">input_ac\n" + Cookies.get("input_ac");

        const finalFormat = input_ac + '\n' + sslstm + '\n' + ssconv + '\n';
        const blob = new Blob([finalFormat], {type : 'text/plain'})
        const url = URL.createObjectURL(blob)
        const a = document.createElement("a");
        a.href = url;
        if (code !== NaN){
            a.download = "secpred_prediction_"+code+" .fasta";
        } else {
            a.download = "secpred_prediction.fasta"; 
        }
        a.click();
        URL.revokeObjectURL(url);
    }

</script>
<button
    class="flex h-[45px] w-[70px] cursor-pointer items-center justify-center rounded-full bg-blue-400 shadow-2xl transition duration-200 hover:scale-110 hover:shadow-black"
    aria-label="Print fast"
    onclick={handleFastaDownload}
>
    <h3 class="text-white">FASTA</h3>
</button>