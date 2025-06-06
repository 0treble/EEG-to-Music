import UnicornPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt
from scipy.stats import entropy
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver-manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def compute_music_features(data_buffer, fs):
    def bandpower(data, fs, band):
        freqs = np.fft.rfftfreq(len(data), 1 / fs)
        fft_vals = np.abs(np.fft.rfft(data)) ** 2
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.sum(fft_vals[idx])

    data = [np.array(ch) for ch in data_buffer]

    # Power in relevant bands
    theta = bandpower(data[0], fs, [4, 7])  # Frontal/central regions for meditation
    alpha = sum(bandpower(data[i], fs, [8, 12]) for i in [1, 2, 3])  # Central/occipital regions
    beta = sum(bandpower(data[i], fs, [13, 30]) for i in [4, 5, 7])  # Frontal/central regions for concentration
    gamma = bandpower(data[6], fs, [30, 50])  # Parieto-occipital regions for emotional processing

    total_power = theta + alpha + beta + gamma + 1e-8
    theta_rel = theta / total_power
    alpha_rel = alpha / total_power
    beta_rel = beta / total_power
    gamma_rel = gamma / total_power

    # Asymmetry (alpha)
    alpha_asym = bandpower(data[1], fs, [8, 12]) - bandpower(data[3], fs, [8, 12])

    # Theta / Beta ratio (related to concentration)
    ratio = theta / (bandpower(data[4], fs, [13, 30]) + 1e-8)

    # Coherence (relevant for emotional processing)
    gamma_po7 = bandpower(data[5], fs, [30, 50])
    gamma_po8 = bandpower(data[7], fs, [30, 50])
    coherence = 1 - abs(gamma_po7 - gamma_po8) / (gamma_po7 + gamma_po8 + 1e-8)

    rms = np.sqrt(np.mean(data[0] ** 2))

    # Cz entropy (for emotional response)
    cz_prob = np.histogram(data[2], bins=50, density=True)[0]
    cz_entropy = entropy(cz_prob + 1e-12)

    # Activity classification based on frequency bands and ratios
    if theta_rel > 0.5 and alpha_rel > 0.3 and beta_rel < 0.1:
        activity = "Meditation"
        mood = "calm and floating"
        tempo = "delicate and slow"
        harmony = "soothing major"
        genre = "Ambient"
        instruments = "synth pads, soft piano, ethereal strings"
    elif beta_rel > 0.4 and alpha_rel < 0.3:
        activity = "Concentration"
        mood = "sharp and focused"
        tempo = "steady, with purpose"
        harmony = "minimal, structured"
        genre = "Jazz"
        instruments = "electric guitar, piano, subtle synths"
    elif gamma_rel > 0.4 and abs(alpha_asym) > 1e9:
        activity = "Emotional Response"
        mood = "dramatic and intense"
        tempo = "building, expressive"
        harmony = "fiery, with tension"
        genre = "Cinematic"
        instruments = "violin, cinematic percussion, electronic elements"
    elif alpha_rel > 0.4 and beta_rel < 0.2:
        activity = "Relaxation"
        mood = "peaceful and serene"
        tempo = "slow and flowing"
        harmony = "gentle, uplifting major"
        genre = "Classical"
        instruments = "grand piano, soft strings, gentle harp"
    elif beta_rel < 0.15 and gamma_rel < 0.2 and alpha_rel > 0.35:
        activity = "Dreamy Relaxation"
        mood = "light and drifting"
        tempo = "slow and drifting"
        harmony = "majestic and wide"
        genre = "New Age"
        instruments = "flute, chimes, light percussion, ambient pads"
    elif gamma_rel > 0.5:
        activity = "High Energy"
        mood = "vibrant and energetic"
        tempo = "fast and pulsating"
        harmony = "bright and powerful"
        genre = "Electronic Dance Music (EDM)"
        instruments = "synth leads, heavy bass, electronic drums, risers"
    elif theta_rel > 0.3 and alpha_rel > 0.3:
        activity = "Focused Relaxation"
        mood = "calm yet alert"
        tempo = "moderate and flowing"
        harmony = "relaxed major"
        genre = "Chillwave"
        instruments = "smooth synths, soft drums, electric guitar"
    elif alpha_asym < 0 and beta_rel > 0.3:
        activity = "Stress"
        mood = "tense and anxious"
        tempo = "fast and unpredictable"
        harmony = "discordant, with tension"
        genre = "Industrial"
        instruments = "distorted guitars, harsh synths, aggressive drums"
    elif beta_rel > 0.2 and gamma_rel > 0.3:
        activity = "Excitement"
        mood = "enthusiastic and lively"
        tempo = "quick and catchy"
        harmony = "major with energy"
        genre = "Pop"
        instruments = "electric guitar, synth bass, upbeat drums, claps"
    elif alpha_rel > 0.35 and beta_rel < 0.25:
        activity = "Euphoria"
        mood = "elated and joyful"
        tempo = "upbeat and fast"
        harmony = "bright, major"
        genre = "Rock"
        instruments = "electric guitar, drums, bass, catchy melodies"
    else:
        activity = "Undefined"
        mood = "neutral, unknown"
        tempo = "ambiguous"
        harmony = "indeterminate"
        genre = "Experimental"
        instruments = "ambient drones, field recordings"

    # Rhythm, arpeggios, balance, width, volume, and melody
    rhythm = "interwoven" if beta_rel > 0.3 else "subtle"
    arp = "glistening" if gamma_rel > 0.4 else "smooth"
    balance = "harmonious" if ratio > 0.5 else "percussive"
    width = "expansive" if coherence > 0.6 else "focused"
    volume = "rich and full" if rms > 1000 else "soft and reserved"
    melody = "flowing and free" if cz_entropy > 2 else "clear and structured"

    prompt = (
        f"\nMental state: {activity} — {mood}\n"
        f"Music: {tempo} tempo, {rhythm} rhythm, {arp} arpeggios, "
        f"{harmony} harmony, {balance} texture, {width} stereo, {volume} volume, {melody} melody.\n"
        f"Genre: {genre} | Instruments: {instruments}\n"
    )
    generate_music_from_prompt(mood, tempo, genre, instruments)

    return prompt


def generate_music_from_prompt(mood, tempo, genre, instruments):
    try:
        options = EdgeOptions()
        options.use_chromium = True

        driver = webdriver.Edge(
            service=EdgeService(EdgeChromiumDriverManager().install()),
            options=options
        )
        wait = WebDriverWait(driver, 30)
        actions = ActionChains(driver)

        # Create the music prompt
        prompt = f"Generate a {mood} {genre} loop with {instruments}, {tempo} tempo"
        print("Music generation prompt:", prompt)

        driver.get(" -- add your MusicGen (or equivalent) API --")

        # Fill textarea via JS
        textarea = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "textarea[data-testid='textbox']")
        ))
        driver.execute_script(
            "arguments[0].value = arguments[1]; arguments[0].dispatchEvent(new Event('input',{ bubbles:true }));",
            textarea, prompt
        )
        time.sleep(0.5)

        # Click Generate
        gen_btn = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "button#component-9")
        ))
        actions.move_to_element(gen_btn).click().perform()

        # Wait for generation
        time.sleep(8)
        print("Submitted prompt:", textarea.get_attribute("value"))

        # Hover & click Play
        play_btn = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'span[aria-label="play-pause-replay-button"]')
        ))
        actions.move_to_element(play_btn).click().perform()
        print("Playing generated music...")

        # Keep the music playing for a while
        time.sleep(15)

    except Exception as e:
        print(f"Error in music generation: {str(e)}")
    finally:
        try:
            driver.quit()
        except:
            pass


def main():
    FrameLength = 1
    DataFile = "data.csv"
    update_interval = 5
    show_plots = False  # Set to True for FFT plots
    fs = UnicornPy.SamplingRate
    eeg_channel_count = 8

    try:
        deviceList = UnicornPy.GetAvailableDevices(True)
        if not deviceList:
            raise Exception("No Unicorn devices found.")
        device = UnicornPy.Unicorn(deviceList[0])
        print(f"\nConnected to Unicorn device: {deviceList[0]}")

        file = open(DataFile, "wb")
        num_channels = device.GetNumberOfAcquiredChannels()

        buffer_len = FrameLength * num_channels * 4
        receiveBuffer = bytearray(buffer_len)

        device.StartAcquisition(False)

        # Battery Level
        device.GetData(FrameLength, receiveBuffer, buffer_len)
        initial_raw = np.frombuffer(receiveBuffer, dtype=np.float32).reshape(FrameLength, num_channels)
        battery = initial_raw[0, -2]
        battery_percent = battery * 100
        print(f"Battery Level: {battery_percent:.0f}%\n")

        # Baseline
        print("Collecting baseline...")
        baseline_data = []
        for _ in range(fs):
            device.GetData(FrameLength, receiveBuffer, buffer_len)
            raw = np.frombuffer(receiveBuffer, dtype=np.float32).reshape(FrameLength, num_channels)
            baseline_data.append(raw[:, :eeg_channel_count])
        baseline = np.mean(np.concatenate(baseline_data), axis=0)
        print("Baseline collected.\n")

        fir_bandpass = firwin(numtaps=401, cutoff=[5, 40], pass_zero=False, fs=fs)
        data_buffer = [[] for _ in range(eeg_channel_count)]

        if show_plots:
            plt.ion()
            fig, axs = plt.subplots(eeg_channel_count, 1, figsize=(12, 3 * eeg_channel_count))
            fft_lines = []
            fft_freqs = np.fft.rfftfreq(fs * update_interval, 1 / fs)
            for ch in range(eeg_channel_count):
                axs[ch].set_xlim([5, 40])
                axs[ch].set_ylim([0, 1])
                axs[ch].set_xlabel("Freq [Hz]")
                axs[ch].set_ylabel("Norm Mag")
                axs[ch].grid(True)
                line, = axs[ch].plot(fft_freqs, np.zeros_like(fft_freqs))
                fft_lines.append(line)
            plt.tight_layout()

        last_update_time = time.time()

        while True:
            device.GetData(FrameLength, receiveBuffer, buffer_len)
            raw = np.frombuffer(receiveBuffer, dtype=np.float32).reshape(FrameLength, num_channels)
            eeg_data = raw[:, :eeg_channel_count] - baseline
            for ch in range(eeg_channel_count):
                data_buffer[ch].extend(eeg_data[:, ch])
                if len(data_buffer[ch]) > fs * update_interval:
                    data_buffer[ch] = data_buffer[ch][-fs * update_interval:]

            np.savetxt(file, eeg_data, delimiter=',', fmt='%.3f', newline='\n')

            if time.time() - last_update_time >= update_interval:
                last_update_time = time.time()

                if show_plots:
                    for ch in range(eeg_channel_count):
                        ch_data = np.array(data_buffer[ch])
                        if len(ch_data) < 1203:
                            continue
                        ch_filtered = filtfilt(fir_bandpass, 1, ch_data)
                        ch_filtered -= np.mean(ch_filtered)
                        fft_vals = np.abs(np.fft.rfft(ch_filtered))
                        fft_vals /= np.max(fft_vals) if np.max(fft_vals) > 0 else 1
                        fft_lines[ch].set_ydata(fft_vals)
                        axs[ch].set_ylim([0, 1])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                else:
                    prompt = compute_music_features(data_buffer, fs)
                    print(prompt)

    finally:
        try:
            device.StopAcquisition()
        except UnicornPy.DeviceException:
            pass
        file.close()
        del device
        print("Disconnected from Unicorn.")


if __name__ == '__main__':
    main()