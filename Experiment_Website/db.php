<?php
    function console_log( $data ){
    // printing at console in .php file
        echo '<script>';
        echo 'console.log('. json_encode( $data ) .')';
        echo '</script>';
    }

    // sql database info 
    $servername = "sql202.epizy.com";
    $username = "epiz_29320404";
    $password = "CrNYzDv8IuI7";
    $dbname = "epiz_29320404_MusicMem";

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }


    if (isset($_POST['index']) and isset($_POST['haveHeard'])) {
        // TODO: modify to insert new row with haveHeard(bool) and audio_idx(int) 
        $sql = "UPDATE test SET lastname='Ron' WHERE id=2";
        if ($conn->query($sql) === TRUE) {
            echo "Record updated successfully";
        } else {
            echo "Error updating record: " . $conn->error;
        }

        // console_log($_POST['action']);
        // switch ($_POST['action']) {
        //     case 'heard':
        //         echo "The heard function is called.";
        //         $sql = "UPDATE test SET lastname='Ron' WHERE id=2";
        //         if ($conn->query($sql) === TRUE) {
        //             echo "Record updated successfully";
        //         } else {
        //             echo "Error updating record: " . $conn->error;
        //         }
        //         break;
        // }
    }



    $conn->close();
?>