<?php

    // sql database info 
    $servername = "sql202.epizy.com";
    $username = "epiz_29320404";
    $password = "CrNYzDv8IuI7";
    $dbname = "epiz_29320404_MusicMem";

    $email = $_POST['email'];
    $audioOrderStr = $_POST['audioOrderStr'];
    $responseStr = $_POST['responseStr'];
    $responsePositionStr = $_POST['responsePositionStr'];

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }


    // TODO: update to experimentD
    $sql = "INSERT INTO experimentData_beta (updateTime, userEmail, audioOrder, userResponse, responsePosition) VALUES (CURRENT_TIMESTAMP, '$email', '$audioOrderStr', '$responseStr', '$responsePositionStr')";

    if ($conn->query($sql) === TRUE) {
        echo "Record updated successfully";
    } else {
        echo "Error updating record: " . $conn->error;
    }
        
    $conn->close();
    
?>
