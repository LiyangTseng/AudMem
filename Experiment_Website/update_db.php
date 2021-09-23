<?php

    // sql database info 
    $servername = "sql202.epizy.com";
    $username = "epiz_29320404";
    $password = "CrNYzDv8IuI7";
    $dbname = "epiz_29320404_MusicMem";

    $startTime = $_POST['startTime'];
    $nowTime = $_POST['nowTime'];
    $email = $_POST['email'];
    $audioOrderStr = $_POST['audioOrderStr'];
    $responseStr = $_POST['responseStr'];
    $responsePositionStr = $_POST['responsePositionStr'];
    $experimentFinished = $_POST['experimentFinished'];

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }


    // TODO: update to experimentData after pilot study
    $sql = "UPDATE experimentData_beta SET updateTime='$nowTime', userResponse='$responseStr', responsePosition='$responsePositionStr', experimentFinished='$experimentFinished' WHERE userEmail='$email' AND startTime='$startTime'";
    
    if ($conn->query($sql) === TRUE) {
        echo "Record updated successfully in update_db.php";
    } else {
        echo "Error updating record: " . $conn->error;
    }
        
    $conn->close();
    
?>
