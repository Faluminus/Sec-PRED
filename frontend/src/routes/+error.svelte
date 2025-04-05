<script>
    import { page } from '$app/state';

    function is404(){
        if(page.status == 404){
            return true
        }
    }

</script>
<div class="w-screen h-screen p-10 pb-[55px] flex flex-row gap-4">
    {#if is404()}
    <div class="flex flex-row items-center justify-center w-full h-full gap-5">
        <h1 class="text-2xl">What are you looking for ... ?</h1>
        <img width="300" src="/Dog Crying Sticker by Sticker Book iOS GIFs.gif">
    </div>
    {/if}
</div>
