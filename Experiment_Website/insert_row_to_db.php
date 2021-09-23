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
    $experimentFinished = $_POST['experimentFinished'];

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }


    // TODO: update to experimentData after pilot study
    $sql = "INSERT INTO experimentData_beta (startTime, updateTime, userEmail, audioOrder, userResponse, responsePosition, experimentFinished) VALUES (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '$email', '$audioOrderStr', '$responseStr', '$responsePositionStr', '$experimentFinished')";

    if ($conn->query($sql) === TRUE) {
        echo "Record inserted successfully in insert_row_to_db.php";
    } else {
        echo "Error inserting record: " . $conn->error;
    }
        
    $conn->close();
    
?>
