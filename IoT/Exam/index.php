
<?php
        echo '<h1> Basic Math Operations + - * /</h1>';

        echo '<h2> https://eheeey.azurewebsites.net/Exam/index.php?x=10&s=+&y=2 </h2>';

        $num1 = $_GET["x"];
        $num2 = $_GET["y"];
        $symbol = $_GET["s"];

        if ($symbol == "-") $sum = $num1 - $num2;
        else if ($symbol == "*") $sum = $num1 * $num2;
        else if ($symbol == "/") $sum = $num1 / $num2;
        else { $symbol = "+"; $sum = $num1 + $num2; }
        //else $sum = "Unknown operation";

        $text = $num1 . " " . $symbol . " " . $num2 . " = " . $sum;

        echo $text;
    ?>
