<script>
    export let code;
    export let species;
    export let time;
</script>
<div class="flex flex-col w-[350px] h-[300px] bg-gray-100 rounded-xl drop-shadow-2xl p-5 font-bold items-center justify-center text-center">
    <div class="flex flex-col h-full w-full">
        <p>Protein code</p>
        <p>{code}</p>
    </div>
    <div class="flex flex-col h-full w-full">
        <p>Species</p>
        <p>{species}</p>
    </div>
    <div class="flex flex-col h-full w-full">
        <p>Time pended</p>
        <p>{time}s</p>
    </div>
</div>