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


const resolvers = {
  Query: {
    aminoAcidSeq: async (_:null, { id }: { id:string }) => {
      try {
        console.log('https://www.rcsb.org/fasta/entry/'+id+'/download')
        const FASTA = fetch('https://www.rcsb.org/fasta/entry/'+id+'/download',{
            method:"GET",
            mode:"cors",
        })
        .then((response)=>{
            return response.text()
        })
        .catch((exception)=>{
            console.log(exception)
        })
        return FASTA
      } catch (error) {

      }
    },
    secStructureSeq: async(_:null,{aminoAcidSeq}:{aminoAcidSeq:string}) =>{
        
    },
    swissModel: async(_:null,{aminoAcidSeq}:{aminoAcidSeq:string}) =>{
        
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