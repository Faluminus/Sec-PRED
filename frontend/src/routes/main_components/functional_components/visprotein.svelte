<script>
    let {protein_code, protein_name, protein_description} = $props()
</script>
<div class="flex flex-col w-[400px] h-40 ">
    <h2 class="text-blue-500 text-[30px] font-bold">{protein_name}</h2>
    <p class="w-full h-full break-words text-[15px] leading-snug">
        {protein_description}
    </p>
</div>