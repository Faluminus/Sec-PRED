<script>
    export let selected;
    export let handleModelChange;
</script>
<form class="w-full">
    <label for="models" class="mb-2 block text-sm font-medium text-gray-900"
        >Pick a model</label
    >
    <select
        bind:value={selected}
        onchange={handleModelChange}
        id="models"
        class="block w-full rounded-lg border border-gray-300 bg-gray-50 p-2.5 text-sm text-gray-900 focus:border-blue-500 focus:ring-blue-500 "
    >
        <option value="LSTM+CNN">LSTM+CNN - 81%</option>
        <option value="CNN">CNN - 74%</option>
        <option value="TRANSFORMER">Transformer</option>
    </select>
</form>