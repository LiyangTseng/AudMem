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

let slot_cnt = 0;
let isPlaying = false;
let updateTimer;

// Create new audio element
let curr_track = document.createElement('audio');

let dir = 'clips/'
// Define the tracks that have to be played, 
// notice the filename should contain alphabet and number to prevent audio.src not found
let track_list = ['clip_C4IuLw86CJ8.wav', 'clip_hjIhCG_nIPA.wav', 'clip_8OGw36lQn4w.wav', 'clip_fadPaKn-xe8.wav', 
'clip_EygNk739nnY.wav', 'clip_AoB8koE95C0.wav', 'clip_GUC6UqXjU44.wav', 'clip_DwgdKSKGMLQ.wav', 'clip_ZpsV5SGa5R4.wav', 
'clip_SZaZU_qi6Xc.wav', 'clip_yPS0PGBsV8I.wav', 'clip_nISI4qF55F4.wav', 'clip_Cd-ur7jyV30.wav', 'clip_9bQY2komrnA.wav', 
'clip_0QN9KLFWn7I.wav', 'clip_j9KKh215HTs.wav', 'clip_ngJ71Np-qFY.wav', 'clip_Et-YdmSo_3A.wav', 'clip_lEHM9HZf0IA.wav', 
'clip_jII5qoCrzYE.wav', 'clip_32MsxqBbe08.wav', 'clip_dpjUrLEgYKI.wav', 'clip_QglaLzo_aPk.wav', 'clip_F_9pZv0wn9Q.wav', 
'clip_tnOuMlDUgu8.wav', 'clip_XJt-4KKCZBA.wav', 'clip_-b7X97M5tDo.wav', 'clip_C5VCGM2J5ls.wav', 'clip_TzhhbYS9EO4.wav', 
'clip_xJt-wTjGkN8.wav', 'clip_7pcZIsJNlAs.wav', 'clip_ht7xflC4APQ.wav', 'clip_WGMa6qh-qFY.wav', 'clip_qkE-siH6Ed4.wav', 
'clip_D6uH2mKJRUA.wav', 'clip_Ny6BAUMZxT4.wav', 'clip_Tfypj4UwvvA.wav', 'clip_XEjLoHdbVeE.wav', 'clip_Z5gvqq3ChII.wav', 
'clip_2UL-1MOlSPw.wav', 'clip_Gq5-cpb5f7E.wav', 'clip_feVUoKhP1mE.wav', 'clip_gCsKlVY8_-Q.wav', 'clip_BXTpedYsjyI.wav', 
'clip_HwcCBnfhsR4.wav', 'clip_xe8-Y1oVQeU.wav', 'clip_D1QQ8UyBFoA.wav', 'clip_o683V8-TAPQ.wav', 'clip_lYxcW8jtFw0.wav', 
'clip_T4stQxboYKM.wav', 'clip_9wgaix00KCE.wav', 'clip_nDjdsytJrf8.wav', 'clip_RhBb77hG0iw.wav', 'clip_vhWoHF9-qfY.wav', 
'clip_p9LLoijPQfg.wav', 'clip_ms5Hfjd-2AI.wav', 'clip_f8NapGGtMvo.wav', 'clip_Mme9REVuidw.wav', 'clip_xJt--P13x3s.wav', 
'clip_xJt-xbN5S3E.wav', 'clip_ORurdsjkJMQ.wav', 'clip_btMBUL8_liA.wav', 'clip_-_P_cD0yimw.wav', 'clip_tnzybViYYHw.wav', 
'clip_BEo0rqOZIng.wav', 'clip_yKEAUgA8OvU.wav', 'clip_QtYDvqJgQqo.wav', 'clip_3TyKyeS3P-Q.wav', 'clip_gwsaElRJI2M.wav', 
'clip_bBcyBylIr40.wav', 'clip_8MAWFIM-2aI.wav', 'clip_2UL-FFrX0xQ.wav', 'clip_ThtO-8h-qfY.wav', 'clip_LKaXY4IdZ40.wav', 
'clip_bd5m12UEHWI.wav', 'clip_zYoVEbs-xe8.wav', 'clip_C90sY_Ht6Ig.wav', 'clip_0Za5671VM-0.wav', 'clip_3ObVN3QQiZ8.wav', 
'clip_z7y6MykrE5s.wav', 'clip_YOKq1VmEbtc.wav', 'clip_1wpJkzCWHcI.wav', 'clip_n4HTXYR-2AI.wav', 'clip_C7u6rtswjCU.wav', 
'clip_t7-OcRIdUu8.wav', 'clip_9arRsdRTTNI.wav', 'clip_GQ5-gYPabd4.wav', 'clip_PYM9NUU9Roc.wav', 'clip_xGPeNN9S0Fg.wav', 
'clip_xjt-NS8R2LA.wav', 'clip_3e2aMWVWecU.wav', 'clip_dl6vG66m1e8.wav', 'clip_8t9RO40-D74.wav', 'clip_D9ffTk7-2aI.wav'];

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
    document.getElementById("toggle-unhrd").checked = true;
    track_index = audioOrder[slotWithBreakOrder[track_counter]];
    clearInterval(updateTimer);
    resetValues();
    curr_track.src =  dir + track_list[track_index];
    curr_track.load();

    // for debug prupose
    track_name.textContent = `No.${track_index+1}`;
    // track_name.textContent = "";
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
  playpause_btn.innerHTML = '<i class="fa fa-pause-circle fa-5x"></i>';
}

function pauseTrack() {
  curr_track.pause();
  isPlaying = false;
  playpause_btn.innerHTML = '<i class="fa fa-play-circle fa-5x"></i>';;
}

function nextTrack() {
  if (slot_cnt%2 === 0){
    // if real audio(music) played in slot
    if (document.getElementById("toggle-alhrd").checked) {
      userResponses.push(-1);
    } else if (document.getElementById("toggle-unhrd").checked) {
      userResponses.push(0);
    } else if (document.getElementById("toggle-hrd").checked) {
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
    window.onbeforeunload = null;
    window.location.href='index.html'
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
  if (email == ""){
    alert("Please refresh page and fill your email!")
  }
  if (!isPlaying) playTrack();
  // else pauseTrack();
}


function submitEmail() {
  email = document.getElementById("userEmail").value;
  console.log(email);
  if (email != ""){
    submitModal.toggle();
    // $('#submitModal').modal('toggle');
  } else {
    alert('You need to fill your email first!');
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

// Load the first track in the tracklist
loadTrack(slot_cnt);