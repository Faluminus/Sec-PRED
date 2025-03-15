<script>
  import AminoAcidText from './AminoAcidText.svelte';
  import HelixVisualization from './HelixVisualization.svelte';
  import CoilVisualization from './CoilVisualization.svelte';
  import SheetVisualization from './SheetVisualization.svelte';
  import { page } from '$app/stores';
  import { derived } from 'svelte/store';
  import { onMount } from 'svelte';

  const rowLen = 80
  const incr = 80
  let {aminoAcid, secondaryStructure} = $props()
  let previousMyProp = secondaryStructure;
  let dataArr = $state()
  let x = $state()
  let y = $state()
  let wrap = (y, insides) => {return `<g transform="translate(0, ${y})" pointer-events="none">${insides}</g>`}
  let clipPath = (id, data) => {return `<g clip-path="url(#${id})">${data}</g>`}
  let defs =  (id) => {return `<defs><g id="icon-ss-Hb-8eaa"><path d="M 7.05 0.5 H 2.35 Q 0 0.5, -7.05 17.5 H -2.35 Q 0 17.5, 7.05 0.5" fill="#ff6600"></path></g><g id="icon-ss-Hf-8eaa"><path d="M -7.05 0.5 H -2.35 Q 0 0.5, 7.05 17.5 H 2.35 Q 0 17.5, -7.05 0.5" fill="#ff9900"></path></g><g id="icon-ss-Hef-8eaa" style="fill: rgb(255, 153, 0);"><path d="M 0 7.2 L 9.4 17.5 H 4.7 Q 2.35 17.5, 0 10.799999999999999"></path></g><g id="icon-ss-Heb-8eaa"><path d="M 11.75 7.2 L 2.35 17.5 H 7.05 Q 9.4 17.5, 11.75 10.799999999999999" fill="#ff6600"></path></g><clipPath id=${id}><rect x="0" y="0" width="940" height="18"></rect></clipPath></defs>`}
  let text = (x, aminoAcid) => {return `<text style="font-size: 13.75px !important;" x="${x}" pointer-events="none" y="18" text-anchor="middle">${aminoAcid}</text>`}
  
  let helix = (helixSymbol, x) => {return `<use xlink:href="${helixSymbol}" transform="translate(${x}, 0)"></use>`}
  let coil = (width, x) => {return `<rect width="${width}" height="3.6" y="7.2" x="${x}" fill="#cc3399"></rect>`}
  let sheet = (width, x) => {return `<rect width="${width}" height="10" y="3.6" x="${x}" fill="#660099"></rect>`}
  let sheetEnd = () => {}

  function rowBlocks(incr){
    y = 0;
    let data = [];
    
    for (let i = 0; i < Math.ceil(aminoAcid.length / rowLen); i++) {
      let rest = rowLen > aminoAcid.length - (i * rowLen) ? aminoAcid.length - (i * rowLen) : rowLen;
      let insides = [getAminoAcids(i * rowLen , i * rowLen + rest), makeVisualization(i * rowLen, i * rowLen + rest, i)];
      data = [...data, insides];
      y += incr;
    }
    return data;
  }

  function getAminoAcids(from, to){
    let insides = [];
    let x = 5.875;
    for(let i = from; i < to; i++){
      insides.push(text(x, aminoAcid[i]));
      x += 11.75;
    }
    return insides;
  }

  function makeVisualization(from, to){
      let visRow = '';
      let insides = []
      for(let i = from; i < to; i++){
          visRow += secondaryStructure[i] === 'H' ? '1' : secondaryStructure[i] === 'B' || secondaryStructure[i] === 'E' ? '2' : secondaryStructure[i] === 'T' || secondaryStructure[i] === 'S' || secondaryStructure[i] === 'C' ? '3' : '0';
      }
      let redundant = 0;
      let temp = '';
      let x = 5.875;
      for(let i = 0; i < to - from; i++){
          if(visRow[i] === '1'){
            if (i % 2 === 0){
              insides.push(helix('#icon-ss-Hb-8eaa', x));         
            } else {
              insides.push(helix('#icon-ss-Hf-8eaa', x));
            }
          }
          if(visRow[i] === '2'){
            insides.push(sheet(11.75,x - 5.875))
          }
          if(visRow[i] === '3'){
            insides.push(coil(11.75,x - 5.875))
          }

          x += 11.75;
          temp = visRow[i]
      }
      
      return insides;
  }

  $effect(() => {
    if (secondaryStructure !== previousMyProp) {
      dataArr = rowBlocks(incr)
      previousMyProp = secondaryStructure;
    }
  });

  onMount(() => {
    dataArr = rowBlocks(incr)
  })

</script>


<svg class='overflow-visible w-full h-auto' viewBox="0 0 940 345" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg">
    {#each dataArr as e, index}
      <g transform="translate(0, {index*incr})">
        <g transform="translate(0, 0)">
          {#each e[0] as text, index}
            <rect 
              class="fill-gray-100 hover:fill-gray-300 transition-colors"
              x={index * 11.75}``
              y="0" 
              z="1"
              width="11.75" 
              height="53" 
              data-tip="true" 
              currentItem="false">
            </rect>
            {@html text}
          {/each}
        </g>
        <g transform="translate(0, 30)">
          {@html defs(index)}
          <g clip-path="url(#{index})">
            {#each e[1] as visual, xa}
              {@html visual}
            {/each}
          </g>
        </g>
      </g>
    {/each}
</svg>
                        