<?php

    // sql database info 
    $servername = "ns29.g-dns.com";
    $username = "musicme1_admin";
    $password = "5MLab3zQI(N*";
    $dbname = "musicme1_database_01";

    $startTime = $_POST['startTime'];
    $nowTime = $_POST['nowTime'];
    $email = $_POST['email'];
    $audioOrderStr = $_POST['audioOrderStr'];
    $responseStr = $_POST['responseStr'];
    $responsePositionStr = $_POST['responsePositionStr'];
    $vigilanceDetail = $_POST['vigilanceDetail'];
    $vigilanceProgress = $_POST['vigilanceProgress'];
    $vigilanceScore = $_POST['vigilanceScore'];
    $token = $_POST['token'];
    $experimentFinished = $_POST['experimentFinished'];

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }


    // TODO: update to experimentData after pilot study
    $sql = "UPDATE experimentData_beta SET updateTime='$nowTime', userResponse='$responseStr', responsePosition='$responsePositionStr', 
            vigilanceDetail='$vigilanceDetail', vigilanceProgress='$vigilanceProgress', vigilanceScore='$vigilanceScore', token='$token', 
            experimentFinished='$experimentFinished' WHERE userEmail='$email' AND startTime='$startTime'";
    
    if ($conn->query($sql) === TRUE) {
        echo "Record updated successfully in update_db.php";
    } else {
        echo "Database Error";
        exit;
        // pass to js to redirect
    }
    
    $conn->close();
    
?>
