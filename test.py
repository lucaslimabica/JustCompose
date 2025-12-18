import time,fluidsynth

SF2=r"assets\FluidR3_GM.sf2"

fs=fluidsynth.Synth()
fs.start(driver="dsound")
sfid=fs.sfload(SF2)

def play_note(ch,prog,note,vel=110,dur=0.25,bank=0):
    fs.program_select(ch,sfid,bank,prog)
    fs.noteon(ch,note,vel);time.sleep(dur);fs.noteoff(ch,note)

# Piano (GM 1 -> program 0)
play_note(0,0,60)
time.sleep(5)
# Bass (GM 34 -> program 33)
play_note(0,33,40)
time.sleep(5)
# Guitar (GM 30 -> program 29)
play_note(0,29,52)
time.sleep(5)
# Synth (GM 81 -> program 80) Lead 1 (square)
play_note(0,80,72)
time.sleep(5)
# Drums: canal 9 (MIDI canal 10), geralmente bank 128
fs.program_select(9,sfid,128,0)
for n in (36,38,42):  # kick, snare, closed hat
    fs.noteon(9,n,127);time.sleep(0.12);fs.noteoff(9,n)
    time.sleep(5)
    
time.sleep(0.2)
fs.delete()
