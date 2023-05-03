<?php

// konstanty s nazvami parametrov
const PARAM_HINT = 'hint';
const PARAM_NAME = 'name';
const PARAM_TYPE = 'type';
const PARAM_TPLACED = 'tPlaced';
const PARAM_HEATER = 'heater';
const PARAM_TREADY = 'tReady';
const PARAM_MINFLUID = 'minFluid';
const PARAM_VALVE = 'valve';
const PARAM_TEMPERATURE = 'temperature';
const PARAM_CREATED_AT = 'createdAt';
const PARAM_UPDATED_AT = 'updatedAt';

// parametre, ktore posiela ESP cez updateParameters.php
static $keys = [
    PARAM_HINT,
    PARAM_NAME,
    PARAM_TYPE,
    PARAM_TPLACED,
    PARAM_HEATER,
    PARAM_TREADY,
    PARAM_MINFLUID,
    PARAM_VALVE,
    PARAM_TEMPERATURE
];

// nazov suboru s datami a aktualny datum
static $filename = "parameters.json";
$datetime = new DateTime();

// definicie hintov - textova sprava o stave ESP
$hints = [
    0 => "",
    1 => "Put the Teapot on it's place.",
    2 => "Not enought Water.",
    3 => "Too much Water.",
    4 => "Press the Start button.",
    5 => "Heating up the Water.",
    6 => "The Water is cooling down.",
    7 => "Adding the Herbs.",
    8 => "Brewing...",
];

// nacitanie dat zo suboru
function getData()
{
    global $filename;
    if (!file_exists($filename)) {
        return null;
    }
    $content = file_get_contents($filename);
    return json_decode($content, JSON_OBJECT_AS_ARRAY);
}

// zapisanie dat do suboru
function setData(array $parametersJson)
{
    global $filename;
    $fp = fopen($filename, 'w');
    fwrite($fp, json_encode($parametersJson));
    fclose($fp);
}
