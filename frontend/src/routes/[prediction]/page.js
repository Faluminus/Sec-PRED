import { error } from '@sveltejs/kit';

export default function load({ params }) {
    fetch('http://127.0.0.1:5000/')
}