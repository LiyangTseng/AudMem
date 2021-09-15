let now_playing = document.querySelector(".now-playing");
let track_name = document.querySelector(".track-name");

let playpause_btn = document.querySelector(".playpause-track");
let next_btn = document.querySelector(".next-track");
let prev_btn = document.querySelector(".prev-track");

let seek_slider = document.querySelector(".seek_slider");
let volume_slider = document.querySelector(".volume_slider");
let curr_time = document.querySelector(".current-time");
let total_duration = document.querySelector(".total-duration");

let email = document.getElementById("userEmail").value;

let alhrd_checked = document.getElementById("toggle-alhrd").checked;
let unhrd_checked = document.getElementById("toggle-unhrd").checked;
let hrd_checked = document.getElementById("toggle-hrd").checked;
let submitBtn = document.getElementById("submitBtn");
submitBtn.style.display = "none";

let slot_cnt = 0;
let isPlaying = false;
let updateTimer;

// Create new audio element
let curr_track = document.createElement('audio');

let dir = 'clips/'
// Define the tracks that will be played, 
// notice the filename should contain alphabet and number to prevent audio.src not found
let track_list = ['clip_w1G3rqVil1s.wav', 'clip_HiPkwl5p1GY.wav', 'clip_ZIiQ1jMqhVM.wav', 'clip_fOYuWVIXgiM.wav', 'clip_hjIhCG_nIPA.wav', 
'clip_VLecCiNKjF0.wav', 'clip_hV-FwW1LgxU.wav', 'clip_9DP0yMwvyWE.wav', 'clip_fadPaKn-xe8.wav', 'clip_UJ5-4GiWNnc.wav', 'clip_EygNk739nnY.wav', 
'clip_AoB8koE95C0.wav', 'clip_GUC6UqXjU44.wav', 'clip_V-ar6MLjy5o.wav', 'clip_DwgdKSKGMLQ.wav', 'clip_SZaZU_qi6Xc.wav', 'clip_nISI4qF55F4.wav', 
'clip_0QN9KLFWn7I.wav', 'clip_j9KKh215HTs.wav', 'clip_ngJ71Np-qFY.wav', 'clip_Et-YdmSo_3A.wav', 'clip_lEHM9HZf0IA.wav', 'clip_jII5qoCrzYE.wav', 
'clip_32MsxqBbe08.wav', 'clip_GQ5-Y-n_Djw.wav', 'clip_gg5sZZPuK_o.wav', 'clip_dpjUrLEgYKI.wav', 'clip_ulj-L3K_Gzs.wav', 'clip_tnOuMlDUgu8.wav', 
'clip_XJt-4KKCZBA.wav', 'clip_C5VCGM2J5ls.wav', 'clip_LGX21vCbzdM.wav', 'clip_TzhhbYS9EO4.wav', 'clip_bI8-2blisUM.wav', 'clip_xJt-wTjGkN8.wav', 
'clip_0M24ENnb4gE.wav', 'clip_7pcZIsJNlAs.wav', 'clip_spcz4MLQKD0.wav', 'clip_ht7xflC4APQ.wav', 'clip_UGTYqTKUl8w.wav', 'clip_dCish9XM5bE.wav', 
'clip_bwQ49N0jVvE.wav', 'clip_2JL_KcEzkqg.wav', 'clip_Yyvo9O8fN-A.wav', 'clip_WGMa6qh-qFY.wav', 'clip_D6uH2mKJRUA.wav', 'clip_Tfypj4UwvvA.wav', 
'clip_Z5gvqq3ChII.wav', 'clip_2F8Kr91wQ0U.wav', 'clip_2UL-1MOlSPw.wav', 'clip_Gq5-cpb5f7E.wav', 'clip_feVUoKhP1mE.wav', 'clip_2ul-hWVNIC0.wav', 
'clip_audaUMOnGxA.wav', 'clip_BXTpedYsjyI.wav', 'clip_HwcCBnfhsR4.wav', 'clip_sjlkxcwhpwA.wav', 'clip_xe8-Y1oVQeU.wav', 'clip_o683V8-TAPQ.wav', 
'clip_lYxcW8jtFw0.wav', 'clip_T4stQxboYKM.wav', 'clip_HN65BuYwGMA.wav', 'clip_RhBb77hG0iw.wav', 'clip_vhWoHF9-qfY.wav', 'clip_v0UvOsCi8mc.wav', 
'clip_XJT-fM4nBJU.wav', 'clip_Mme9REVuidw.wav', 'clip_xJt--P13x3s.wav', 'clip_ORurdsjkJMQ.wav', 'clip_btMBUL8_liA.wav', 'clip_-_P_cD0yimw.wav', 
'clip_BEo0rqOZIng.wav', 'clip_haCay85cpvo.wav', 'clip_gwsaElRJI2M.wav', 'clip_0aC-jOKuBFE.wav', 'clip_8MAWFIM-2aI.wav', 'clip_dtOv6WvJ44w.wav', 
'clip_ThtO-8h-qfY.wav', 'clip_bd5m12UEHWI.wav', 'clip_C90sY_Ht6Ig.wav', 'clip_19Q9l85Feqw.wav', 'clip_3ObVN3QQiZ8.wav', 'clip_z7y6MykrE5s.wav', 
'clip_lfy8tbM0q18.wav', 'clip_SubIr_Fyp4M.wav', 'clip_YOKq1VmEbtc.wav', 'clip_1wpJkzCWHcI.wav', 'clip_n4HTXYR-2AI.wav', 'clip_C7u6rtswjCU.wav', 
'clip_PYM9NUU9Roc.wav', 'clip_xjt-NS8R2LA.wav', 'clip_dl6vG66m1e8.wav', 'clip_yBzk2xXE9yg.wav', 'clip_D9ffTk7-2aI.wav'];

// var submitModal = new bootstrap.Modal(document.getElementById("submitModal"));
// submitModal.show();
var submitModal = new bootstrap.Modal(document.getElementById("submitModal"), {
  keyboard: false,
  backdrop: 'static'
});
submitModal.show();
// $(window).on('load', function() {
//     $('#submitModal').modal('show');
// });


function shuffle(array) {
  var currentIndex = array.length,  randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
    }
    
  return array;
}

let audioOrder = [...Array(94).keys()];
// order of every audio
let slotOrder = [0, 1, 2, 1, 3, 4, 0, 5, 6, 7, 3, 8, 9, 10, 5, 11, 12, 7, 13, 12, 14, 10, 15, 16, 
  17, 18, 19, 20, 21, 22, 23, 15, 24, 25, 24, 18, 26, 17, 27, 28, 29, 23, 30, 31, 27, 32, 33, 34, 29, 
  35, 36, 32, 37, 38, 21, 39, 40, 9, 39, 41, 42, 40, 43, 44, 20, 34, 45, 26, 46, 47, 43, 48, 49, 47, 
  50, 51, 14, 48, 52, 45, 53, 50, 54, 55, 56, 57, 51, 58, 59, 55, 28, 60, 61, 62, 61, 63, 58, 64, 62, 
  65, 66, 60, 67, 68, 65, 69, 70, 66, 4, 71, 72, 44, 11, 73, 52, 19, 74, 75, 72, 74, 76, 70, 38, 73, 77, 
  67, 64, 56, 78, 13, 76, 75, 37, 79, 80, 77, 80, 78, 30, 81, 57, 53, 82, 83, 84, 85, 86, 84, 87, 22, 82,
   81, 88, 83, 89, 90, 91, 86, 91, 92, 93, 89];
// order of every audio and break (-1 for 5s break, -2 for 3m break)
let slotWithBreakOrder = Array(323).fill(-1);
for (let i = 0; i < slotWithBreakOrder.length; i++) {
  if (i%2 === 0) {
      // insert break between audios
      slotWithBreakOrder[i] = slotOrder[i/2];
  }    
}
slotWithBreakOrder[54*2-1] = -2;
slotWithBreakOrder[108*2-1] = -2;

let userResponses = [];


function loadTrack(track_counter) {
  if (slotWithBreakOrder[track_counter] === -1) {
    // 5s break
    clearInterval(updateTimer);
    resetValues();
    curr_track.src = "pink_noise_5s.wav";
    curr_track.load();
    
    track_name.textContent = `5 Seconds Break`;
    now_playing.textContent = "";
    $('label[for="toggle-hrd"]').hide();
    $('label[for="toggle-unhrd"]').hide();
    $('label[for="toggle-alhrd"]').hide();
  }
  else if (slotWithBreakOrder[track_counter] === -2) {
    // 3m break
    clearInterval(updateTimer);
    resetValues();
    curr_track.src = "pink_noise_3m.wav";
    curr_track.load();
    
    track_name.textContent = `3 Minutes Break`;
    now_playing.textContent = "";
    $('label[for="toggle-hrd"]').hide();
    $('label[for="toggle-unhrd"]').hide();
    $('label[for="toggle-alhrd"]').hide();
  }
  else {
    // ordinay audio
    $('label[for="toggle-hrd"]').show();
    $('label[for="toggle-unhrd"]').show();
    $('label[for="toggle-alhrd"]').show();
    document.getElementById("toggle-alhrd").checked = false;
    document.getElementById("toggle-unhrd").checked = false;
    document.getElementById("toggle-hrd").checked = false;
    track_index = audioOrder[slotWithBreakOrder[track_counter]];
    clearInterval(updateTimer);
    resetValues();
    curr_track.src =  dir + track_list[track_index];
    curr_track.load();

    // for debug prupose
    // track_name.textContent = `No.${track_index+1}`;
    track_name.textContent = "";
    now_playing.textContent = "PLAYING " + (track_counter/2 + 1) + " OF " + slotOrder.length;
    
  
  }
  updateTimer = setInterval(seekUpdate, 1000);
  curr_track.addEventListener("ended", nextTrack);
}

function resetValues() {
  curr_time.textContent = "00:00";
  total_duration.textContent = "00:00";
  seek_slider.value = 0;
}

function playTrack() {
  curr_track.play();
  isPlaying = true;
}

function pauseTrack() {
  curr_track.pause();
  isPlaying = false;
  playpause_btn.innerHTML = '<i class="fa fa-play-circle fa-5x"></i>';;
}

function getUserResponse() {
  return new Promise((resolve, reject) => {
    (function waitForResponse(){
      if (document.getElementById("toggle-alhrd").checked || document.getElementById("toggle-unhrd").checked || document.getElementById("toggle-hrd").checked) {
        $('label[for="toggle-alhrd"]').removeClass("glowed");
        $('label[for="toggle-unhrd"]').removeClass("glowed");
        $('label[for="toggle-hrd"]').removeClass("glowed");
        return resolve();
      }
      else {
        setTimeout(waitForResponse, 30);
        $('label[for="toggle-alhrd"]').addClass("glowed");
        $('label[for="toggle-unhrd"]').addClass("glowed");
        $('label[for="toggle-hrd"]').addClass("glowed");  
      }
    })();
  })  
}
async function nextTrack() {
  if (slot_cnt%2 === 0){ // if real audio(music) played in slot
    
    await getUserResponse();
  
    if (document.getElementById("toggle-alhrd").checked) {
      // already heard: -1
      userResponses.push(-1);
    } else if (document.getElementById("toggle-unhrd").checked) {
      // unheard: 0
      userResponses.push(0);
    } else if (document.getElementById("toggle-hrd").checked) {
      // unheard: 1
      userResponses.push(1);
    }

  }
  if (slot_cnt < slotWithBreakOrder.length-1) {
    slot_cnt += 1;    
  }  
  else {
    // experiment over
    saveToDB();
    alert('Thank you for your time, this is the end of experiment \nRedirecting to homepage...');
    (() => {
      setInterval(() => {
        window.onbeforeunload = null;
        window.location.href='index.html';            
      }, 1000);
    })();
  }
  loadTrack(slot_cnt);
  playTrack();
}

function prevTrack() {
  if (slot_cnt > 0)
    slot_cnt -= 1;
  else slot_cnt = track_list.length;
  loadTrack(slot_cnt);
  playTrack();
}

function setVolume() {
  // TODO: make work instaneously (when mouse still pressing)
  curr_track.volume = volume_slider.value / 100;
}

function seekUpdate() {
  let seekPosition = 0;

  if (!isNaN(curr_track.duration)) {
    seekPosition = curr_track.currentTime * (100 / curr_track.duration);

    seek_slider.value = seekPosition;

    let currentMinutes = Math.floor(curr_track.currentTime / 60);
    let currentSeconds = Math.floor(curr_track.currentTime - currentMinutes * 60);
    let durationMinutes = Math.floor(curr_track.duration / 60);
    let durationSeconds = Math.floor(curr_track.duration - durationMinutes * 60);

    if (currentSeconds < 10) { currentSeconds = "0" + currentSeconds; }
    if (durationSeconds < 10) { durationSeconds = "0" + durationSeconds; }
    if (currentMinutes < 10) { currentMinutes = "0" + currentMinutes; }
    if (durationMinutes < 10) { durationMinutes = "0" + durationMinutes; }

    curr_time.textContent = currentMinutes + ":" + currentSeconds;
    total_duration.textContent = durationMinutes + ":" + durationSeconds;
  }
}


function playpauseTrack() {
  if (!isPlaying) {
    // Load the first track in the tracklist
    loadTrack(slot_cnt);
    playTrack();
    playpause_btn.style.display = 'none';
    let hour = 1, min = 15, sec = 0;
    let totalTimeSec = 75*60;
    let timeLeftSec = 75*60;
    
    let timerStart = setInterval(function(){
      timeLeftSec --;
      if (timeLeftSec > 0){
        $('.progress-bar').css('width', timeLeftSec/totalTimeSec*100+'%');
        hour = Math.floor(timeLeftSec/60/60);
        min = Math.floor((timeLeftSec/60)%60);
        sec = timeLeftSec%60;
        $('.countDown').text("Time Left " + String(hour).padStart(1, '0') + ":" + String(min).padStart(2, '0') + ":"+ String(sec).padStart(2, '0'));

      } else {
        clearInterval(timerStart);
        alert("Time's up! You failed to complete the experiment on time!");
        window.onbeforeunload = null;
        window.location.href='index.html';
      }
      
    }, 1000);
  }
  // else pauseTrack();
}

function progress(timeleft, timetotal, $element) {
  let progressBarWidth = timeleft * $element.width() / timetotal;
  
  // $element.find('div').animate({ width: progressBarWidth }, 500).html(String(hour).padStart(2, '0') + ":" + String(min).padStart(2, '0') + ":"+ String(sec).padStart(2, '0'));
  $('.bar').animate({ width: progressBarWidth }, 500);
  $('.countDown').text(String(hour).padStart(2, '0') + ":" + String(min).padStart(2, '0') + ":"+ String(sec).padStart(2, '0'));
  if(timeleft > 0) {
      setTimeout(function() {
          // progress(timeleft - 1, timetotal, $element);
          timeleft = timeleft- 1
          progress();
      }, 1000);
  } else {
    alert("Time's up! You failed to complete the experiment on time!");
    window.onbeforeunload = null;
    window.location.href='index.html';
  }
};


function submitEmail() {
  email = document.getElementById("userEmail").value;
  agreed = document.getElementById("agreeCheck").checked
  if (email != "" && agreed == true){
    console.log('log in as', email);
    submitModal.toggle();
    } 
}

function saveToDB() {
  // save to db as strings
  let audioOrderStr = audioOrder.join();
  let responseStr = userResponses.join();
  data = {"email": email, "audioOrderStr": audioOrderStr, "responseStr": responseStr};
  console.log(data);
  $.post('db.php', data, function (response) {
    console.log(response);
    console.log("user data successfully saved to db !!");
  });
  
}

// popup warning about leaving experiment
window.onbeforeunload = function(){
  return 'Experiment data will be lost';
};

// ===============================
audioOrder = shuffle(audioOrder);
$('label[for="toggle-hrd"]').hide();
$('label[for="toggle-unhrd"]').hide();
$('label[for="toggle-alhrd"]').hide();
track_name.textContent = "";


