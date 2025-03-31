export async function fetchPDB(){
    try {
        const FASTA = await fetch('https://www.rcsb.org/fasta/entry/'+input_search+'/download',{
            method:"GET",
            mode:"cors",
            headers: {
                'Access-Control-Allow-Origin':'*'
            }
        })
        .then(async (response)=>{
            const data_splited = (await response.text()).split('\n');
            for(let row of data_splited){
              if(row.split('')[0] != '>'){
                return row;
              }
            }
            console.log(data_splited)
        })
        .catch((exception)=>{
            console.log(exception)
        })
        input_ac = FASTA
        console.log(input_ac)
      } catch (error) {

      }
}