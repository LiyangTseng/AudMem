let now_playing = document.querySelector(".now-playing");
let track_name = document.querySelector(".track-name");

let playpause_btn = document.querySelector(".playpause-track");
let next_btn = document.querySelector(".next-track");
let prev_btn = document.querySelector(".prev-track");

let seek_slider = document.querySelector(".seek_slider");
let volume_slider = document.querySelector(".volume_slider");
let curr_time = document.querySelector(".current-time");
let total_duration = document.querySelector(".total-duration");

let slot_cnt = 90;
let isPlaying = false;
let updateTimer;

// Create new audio element
let curr_track = document.createElement('audio');

let dir = 'clips/'
// Define the tracks that have to be played
let track_list = ['clip_dl6vG66m1e8_Ilkay Sencan & Mert Hakan - Let Me.wav', 
                  'clip_Et-YdmSo_3A_MC Magal e MC G15 - Menor Periferia (GR6 Filmes) DJ Guil Beats.wav', 
                  'clip_yPS0PGBsV8I_Radio Killer - Be Free [Official video HD].wav', 
                  'clip_XJt-4KKCZBA_George X - Nahuati (PHCK Remix) [Sound Avenue].wav', 
                  'clip_bd5m12UEHWI_Duo Anggrek - Goyang Nasi Padang (Official Music Video NAGASWARA) #goyangnasipdg.wav', 
                  'clip_C7u6rtswjCU_Odhani – Made In China _ Rajkummar Rao & Mouni Roy _  Neha Kakkar & Darshan Raval _ Sachin – Jigar.wav', 
                  'clip_D9ffTk7-2aI_Base de Rap _ Boom Bap _ Agresivo _ Underground _ Freestyle Beat _ Hip Hop Instrumental.wav', 
                  'clip_ZpsV5SGa5R4_Lost Sky - Need You [NCS Release].wav', 'clip_o683V8-TAPQ_Opozit - X-oyu-L (Official VIDEO).wav', 
                  'clip_j9KKh215HTs_TiiwTiiw - Te amo feat Blanka & Sky (Selfie Algerian Cover).wav', 
                  'clip_qkE-siH6Ed4_Huz Kai - Ghost (Official Music Video).wav', 
                  'clip_xGPeNN9S0Fg_One Direction - Little Things.wav', 
                  'clip_-b7X97M5tDo_Trixtor - Runaway (ft. Holly Drummond).wav', 
                  'clip_p9LLoijPQfg_TREASURE - ‘MY TREASURE’ M_V.wav', 
                  'clip_TzhhbYS9EO4_Boubkeur ... Huz Huz  ( Audio Officiel ).wav',
                  'clip_bBcyBylIr40_Zedd & Aloe Blacc & Grey - Candyman.wav', 
                  'clip_F_9pZv0wn9Q_[달의 연인 - 보보경심 려 OST Part 6] 에픽하이 (EPIK HIGH) - 내 마음이 들리나요 Can You Hear My Heart (Feat. 이하이 LEE HI).wav', 
                  'clip_vhWoHF9-qfY_Parry Gripp - Nice Big Monster Truck [Official 4K Video].wav', 
                  'clip_Z5gvqq3ChII_Дома не сиди.wav', 
                  'clip_BXTpedYsjyI_MC G15 e MC Bruninho - A Distância ta Maltratando (GR6 Filmes) DJ DG e Batidão Stronda.wav', 
                  'clip_nISI4qF55F4_T-Bone - Crazy Hispanic.wav', 
                  'clip_2UL-1MOlSPw_Nativ - MVZ Vol. 2 - 07 Slick Rick (feat. Dawill).wav', 
                  'clip_0Za5671VM-0_Setec - Water Or Concrete.wav', 
                  'clip_32MsxqBbe08_Merseil - Hi Marimba (Original).wav', 
                  "clip_SZaZU_qi6Xc_Carole Samaha - Esma'ny (Official Music Video) _ كاروول سماحة - إسمعني - الكليب الرسمي.wav", 
                  'clip_Ny6BAUMZxT4_Huz - Early Morning Rain.wav', "clip_C90sY_Ht6Ig_Calvin Markus - Emma's Eyes ('Daredevil - Season 3 Date Announcement' Trailer Music).wav", 
                  "clip_zYoVEbs-xe8_Thrill Collins 'Black or White'.wav", 
                  "clip_tnOuMlDUgu8_Justin Sane & Huz - It's Not Over (Official Music Video).wav",
                  'clip_LKaXY4IdZ40_Whitney Houston, Mariah Carey - When You Believe (Official HD Video).wav', 
                  'clip_C5VCGM2J5ls_وائل جسار - بحبك مش هقول تاني _ Wael Jassar - Ba7ebek Mesh Ha2oul Tany.wav', 
                  'clip_8OGw36lQn4w_Mazigh ...  Huz Huz A Yamina.wav', 
                  'clip_Mme9REVuidw_Zouhair Bahaoui - Hasta Luego ft TiiwTiiw & CHK #DreamTiiw2k17.wav',
                  'clip_btMBUL8_liA_La foudre.wav', 
                  'clip_nDjdsytJrf8_Simple Plan - Kiss Me Like Nobody’s Watching (Lyrics).wav', 
                  'clip_3ObVN3QQiZ8_Nancy Ajram - Oul Tani Keda (Official Music Video) _ نانسي عجرم - قول تاني كده.wav', 
                  'clip_GUC6UqXjU44_Kamel Igman ... Ayen A Thayr iw , Slam fella wen , Huz Huz.wav', 
                  'clip_RhBb77hG0iw_Nerga3 Tany - Hamoot Wa Arga3 _ Tamer Hosny - نرجع تاني - هموت و ارجع.wav', 
                  'clip_xJt--P13x3s_04. Stiinte Oculte - Sa Cada Ploaia.wav', 
                  'clip_T4stQxboYKM_NASSIMA AIT AMI & YACINE YEFSAH - HUZ A YAMINA.wav', 
                  'clip_dpjUrLEgYKI_Justin Sane & Huz - Days With You (Official Music Video).wav', 
                  'clip_xJt-xbN5S3E_Yani ft. Stian - WICKED VISION 2012.wav', 
                  'clip_yKEAUgA8OvU_M. Pokora, Dadju - Si on disait (Clip officiel).wav', 
                  'clip_EygNk739nnY_MC Arthur - XJ6 (Love Funk) DJ GH.wav', 
                  'clip_Cd-ur7jyV30_(FREE) Old School x 90s x Joey Bada$$ Boom Bap Type Beat [2021] - Robin Hood.wav', 
                  'clip_QtYDvqJgQqo_Coone ft. David Spekter - Faye (Official Lyric Video).wav', 
                  'clip_xe8-Y1oVQeU_A Revolta Social “BP” ⛔ 🇧🇷 (ARS PRODUÇÕES).wav', 
                  'clip_D6uH2mKJRUA_Huz - Curious (Original Mix).wav', 
                  'clip_tnzybViYYHw_MC G15 - Cara Bacana (KondZilla).wav', 
                  'clip_9arRsdRTTNI_Trance - Two hearts together.wav', 
                  'clip_WGMa6qh-qFY_SOLO VOY - M R F.wav', 
                  'clip_7pcZIsJNlAs_Didine Canon 16 ft. Larbi Maestro - Tir Ellil طير الليل.wav', 
                  'clip_feVUoKhP1mE_Izuku - jamais vu (Clip Officiel).wav', 
                  'clip_3e2aMWVWecU_Phil Smooth - FAKE LOVE (lyrics_lyrics video).wav', 
                  'clip_fadPaKn-xe8_Dime - Refill (VIDEOCLIP OFICIAL).wav', 
                  'clip_9bQY2komrnA_I Give You My Heart - Hillsong Worship & Delirious.wav', 
                  'clip_f8NapGGtMvo_TiiwTiiw - DAWDAW ft Cheb Nadir, Blanka & Sky (DJ La Meche).wav', 
                  "clip_-_P_cD0yimw_JO1｜'Born To Be Wild' Official MV.wav", 
                  'clip_1wpJkzCWHcI_Nancy Ajram - Oul Hansak (Official Audio) _ نانسي عجرم - قول هنساك.wav', 
                  "clip_3TyKyeS3P-Q_JYJ 'BACK SEAT' M_V.wav", 
                  'clip_Tfypj4UwvvA_Going Deeper feat. Davis Mallory - Believe  _ #GANGSTERMUSIC.wav', 
                  'clip_GQ5-gYPabd4_Stargazing.wav', 'clip_t7-OcRIdUu8_Edgardo Donato - Se Va La Vida.wav', 
                  'clip_xjt-NS8R2LA_Garcia - Anseio (Prod. Pacific).wav', 
                  'clip_ht7xflC4APQ_LIL X - Think  ( prod.pluto ).wav', 
                  'clip_D1QQ8UyBFoA_Gene Krupa - Ball of Fire.wav', 
                  'clip_gCsKlVY8_-Q_Pezet - Mamy ten styl.wav', 
                  'clip_YOKq1VmEbtc_GQ-I do love you.wav', 
                  'clip_AoB8koE95C0_Quando Rondo - End Of Story (Official Audio).wav', 
                  'clip_8MAWFIM-2aI_Enamorado.wav', 
                  "clip_2UL-FFrX0xQ_'그 여름에' 라이브 버스킹♬ 최솜 with 만렙뮤즈 MLMUSE.wav", 
                  'clip_n4HTXYR-2AI_Sadhguru - D Neutrons Music _ Turban Trap.wav', 
                  'clip_BEo0rqOZIng_Huz Kai - On & On (Official Music Video).wav', 
                  'clip_PYM9NUU9Roc_โอเคป่ะ (Yes or No) feat. นุช วิลาวัลย์ อาร์ สยาม  - Flame เฟลม _ Official MV.wav', 
                  'clip_gwsaElRJI2M_Kayblack - A Cunhada (prod. Wall Hein e DJ RB).wav', 
                  'clip_8t9RO40-D74_De Underjordiske - Trold.wav', 
                  'clip_ms5Hfjd-2AI_Os Pontos Negros - Queda e Ascensão.wav', 
                  'clip_z7y6MykrE5s_MC G15 - Eu Falei Pra Elas (KondZilla).wav', 
                  'clip_XEjLoHdbVeE_ABBA - Gimme! Gimme! Gimme! (A Man After Midnight).wav', 
                  'clip_0QN9KLFWn7I_Kozah - Paradox [NCS Release].wav', 
                  'clip_DwgdKSKGMLQ_HUZ - Panamera.wav', 
                  "clip_HwcCBnfhsR4_DAY6 'You make Me' M_V.wav", 
                  'clip_ngJ71Np-qFY_Orli - Internet Explorer (Prod. Amaro JNT).wav', 
                  'clip_hjIhCG_nIPA_MC G15 - Cara Bacana (Lyric Video) Jorgin Deejhay.wav', 
                  'clip_lEHM9HZf0IA_Diamond Eyes - Flutter [NCS Release].wav', 
                  'clip_jII5qoCrzYE_MJ (써니사이드) - +2kg (feat. 효빈) [Lyric Video].wav', 
                  "clip_C4IuLw86CJ8_Huz 'Hardest Thing' (Official Vid).wav", 
                  'clip_QglaLzo_aPk_Julius Dreisig & Zeus X Crona - Invisible [NCS Release].wav', 
                  'clip_ORurdsjkJMQ_Huz Kai - Crush (Visualizer).wav', 
                  'clip_9wgaix00KCE_ป่ะล่ะ (PALA) - POKMINDSET [Official MV].wav', 
                  'clip_Gq5-cpb5f7E_Leoparte - Floodlands (2015).wav', 
                  'clip_xJt-wTjGkN8_[M_V] 도시 of 레인보우 페이퍼 (Dosi of Rainbow paper) - 겨울향기 (Memories of winter) (Official Music Video).wav', 
                  'clip_ThtO-8h-qfY_Briango - Bachata de Amor  Lyrics.wav', 
                  'clip_lYxcW8jtFw0_MC G15 - Deu Onda (KondZilla).wav'];

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

let slotOrder = [0, 1, 2, 1, 3, 4, 0, 5, 6, 7, 3, 8, 9, 10, 5, 11, 12, 7, 13, 12, 14, 10, 15, 16, 
  17, 18, 19, 20, 21, 22, 23, 15, 24, 25, 24, 18, 26, 17, 27, 28, 29, 23, 30, 31, 27, 32, 33, 34, 29, 
  35, 36, 32, 37, 38, 21, 39, 40, 9, 39, 41, 42, 40, 43, 44, 20, 34, 45, 26, 46, 47, 43, 48, 49, 47, 
  50, 51, 14, 48, 52, 45, 53, 50, 54, 55, 56, 57, 51, 58, 59, 55, 28, 60, 61, 62, 61, 63, 58, 64, 62, 
  65, 66, 60, 67, 68, 65, 69, 70, 66, 4, 71, 72, 44, 11, 73, 52, 19, 74, 75, 72, 74, 76, 70, 38, 73, 77, 
  67, 64, 56, 78, 13, 76, 75, 37, 79, 80, 77, 80, 78, 30, 81, 57, 53, 82, 83, 84, 85, 86, 84, 87, 22, 82,
   81, 88, 83, 89, 90, 91, 86, 91, 92, 93, 89];

function loadTrack(track_counter) {
  track_index = audioOrder[slotOrder[track_counter]];
  clearInterval(updateTimer);
  resetValues();
  curr_track.src =  dir + track_list[track_index];
  curr_track.load();
  
  track_name.textContent = `No.${track_index+1} ==== ${track_list[track_index].slice(17, -4)}`;
  now_playing.textContent = "PLAYING " + (track_counter + 1) + " OF " + slotOrder.length;
  
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
  if (slot_cnt < slotOrder.length - 1) {
    slot_cnt += 1;
  }  
  else {
    slot_cnt = 0;
    alert('end of experiment')
    // TODO: terminate experiment
  }
  if (slot_cnt == 54 || slot_cnt == 108){
    alert(`reach slot_cnt ${slot_cnt}, need to implement resting timer`);
    // TODO: timer between stages
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

function startExperiment() {
  if (!isPlaying) playTrack();
  else pauseTrack();
}

function heard(){
  // when "Heard" button pressed
  data =  {'action': 'heard'};
  console.log('entering heard function');
  $.post('db.php', data, function (response) {
      // Response div goes here.
      alert("action performed successfully");
  });
};

// ===============================
// audioOrder = shuffle(audioOrder);

// Load the first track in the tracklist
loadTrack(slot_cnt);

startExperiment();
