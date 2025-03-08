<script>
    import AminoAcidText from './AminoAcidText.svelte';
    import HelixVisualization from './HelixVisualization.svelte';
    import CoilVisualization from './CoilVisualization.svelte';
    import SheetVisualization from './SheetVisualization.svelte';
	import { page } from '$app/stores';
	import { derived } from 'svelte/store';
	import { onMount } from 'svelte';

    const rowLen = 50
    let {aminoAcid, secondaryStructure} = $props()
    
    let dataArr = $state()
	let x = $state()
	let y = $state()
	let wrap = (y, insides) => {return `<g transform="translate(0, ${y})">${insides}</g>`}
    let clipPath = (id, data) => {return `<g clip-path="url(#${id})">${data}</g>`}
    let defs =  (id) => {return `<defs><g id="icon-ss-Hb-8eaa"><path d="M 7.05 0.5 H 2.35 Q 0 0.5, -7.05 17.5 H -2.35 Q 0 17.5, 7.05 0.5" fill="#ff6600"></path></g><g id="icon-ss-Hf-8eaa"><path d="M -7.05 0.5 H -2.35 Q 0 0.5, 7.05 17.5 H 2.35 Q 0 17.5, -7.05 0.5" fill="#ff9900"></path></g><g id="icon-ss-Hef-8eaa" style="fill: rgb(255, 153, 0);"><path d="M 0 7.2 L 9.4 17.5 H 4.7 Q 2.35 17.5, 0 10.799999999999999"></path></g><g id="icon-ss-Heb-8eaa"><path d="M 11.75 7.2 L 2.35 17.5 H 7.05 Q 9.4 17.5, 11.75 10.799999999999999" fill="#ff6600"></path></g><clipPath id=${id}><rect x="0" y="0" width="940" height="18"></rect></clipPath></defs>`}
  
  
    function rowBlocks(incr){
      y = 0;
      let data = [];
      
      for (let i = 0; i < Math.ceil(aminoAcid.length / rowLen); i++) {
        let rest = rowLen > aminoAcid.length - (i * rowLen) ? aminoAcid.length - (i * rowLen) : rowLen;
        let insides = wrap(0,getAminoAcids(i * rowLen , i * rowLen + rest)) + wrap(23,makeVisualization(i * rowLen, i * rowLen + rest, i));
        data = [...data, wrap(y, insides)];
        y += incr;
      }
      return data;
    }
  
    function getAminoAcids(from, to){
      let insides = '';
      let x = 5.875;
      for(let i = from; i < to; i++){
        insides += `<AminoAcidText x="${x}" aminoAcid="${aminoAcid[i]}" />`;
        x += 11.75;
      }
      return insides;
    }
  
    function makeVisualization(from, to, clip){
        let visRow = '';
        for(let i = from; i < to; i++){
            visRow += secondaryStructure[i] === 'H' ? '1' : secondaryStructure[i] === 'B' || secondaryStructure[i] === 'E' ? '2' : secondaryStructure[i] === 'T' || secondaryStructure[i] === 'S' || secondaryStructure[i] === 'C' ? '3' : '0';
        }
        
        for(let i = from; i < to; i++){
            if(i == '1'){

            }
        }

        return defs(clip) + clipPath(clip, visRow);
    }

    onMount(() =>{
       dataArr = rowBlocks(115)
    }) 
</script>


<svg class='overflow-visible w-full h-auto' viewBox="0 0 940 345" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg">
    <AminoAcidText x="12" aminoAcid="XAXAXAXA" />
    {#each dataArr as e}
        {@html e}
    {/each}
</svg>
                        

