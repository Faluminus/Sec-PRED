'use server'
import { startServerAndCreateNextHandler } from "@as-integrations/next";
import { ApolloServer } from "@apollo/server";
import { NextRequest } from "next/server";
import { gql } from "graphql-tag";



const token = "d3c6a4c04dcea0c54025753e4410b8e6c2276d39"

const typeDefs = gql`
  type Query {
    aminoAcidSeq(id: String!):  String
    secStructureSeq(aminoAcidSeq: String!): String
    swissModel(aminoAcidSeq:String!): String
  }
`;

interface SecStructureResponse {
  output: string;
}

const resolvers = {
  Query: {
    aminoAcidSeq: async (_:null, { id }: { id:string }) => {
      try {
        const FASTA = fetch('https://www.rcsb.org/fasta/entry/'+id+'/download',{
            method:"GET",
            mode:"cors",
        })
        .then(async (response)=>{
            const data_splited = (await response.text()).split('\n');
            for(let row of data_splited){
              if(row.split('')[0] != '>'){
                return row;
              }
            }
        })
        .catch((exception)=>{
            console.log(exception)
        })
        return FASTA
      } catch (error) {

      }
    },
    secStructureSeq: async (_: null, { aminoAcidSeq }: { aminoAcidSeq: string }) => {
      try {
          const response = await fetch('http://127.0.0.1:5000/predict-structure', {
              method: "POST",
              headers: {
                  'Content-Type': 'application/json'
              },
              body: JSON.stringify({ "AC": aminoAcidSeq }),
          });
          const secStructureSeq = await response.json() as SecStructureResponse;
          return secStructureSeq.output;
      } catch (error) {
          console.error(error);
          return null;
      }
    },
    swissModel: (_:null,{aminoAcidSeq}:{aminoAcidSeq:string}) =>{
        const headers = new Headers()
        headers.set('Authorization',token)

        const requestOptions = {
          method:'POST',
          headers:headers,
          body: JSON.stringify(
            {
              "target_sequences": aminoAcidSeq
            }
          )
        }
        try {
        const StartModeling  = fetch('https://swissmodel.expasy.org/automodel',requestOptions)
        .then((response) => {
          return response.text()
        })
        } catch (error) {
          
        }
    },
  },
};

const server = new ApolloServer({
    typeDefs,
    resolvers,
});

const handler = startServerAndCreateNextHandler<NextRequest>(server, {
    context: async req => ({ req }),
});


export { handler as GET, handler as POST };