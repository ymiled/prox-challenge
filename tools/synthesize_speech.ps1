param(
    [Parameter(Mandatory = $true)]
    [string]$Text,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

try {
    $voices = $synth.GetInstalledVoices() | Where-Object { $_.Enabled }
    $preferred = $voices | Where-Object { $_.VoiceInfo.Name -eq 'Microsoft Zira Desktop' } | Select-Object -First 1
    if (-not $preferred) {
        $preferred = $voices | Where-Object { $_.VoiceInfo.Name -eq 'Microsoft David Desktop' } | Select-Object -First 1
    }
    if (-not $preferred) {
        $preferred = $voices | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-US*' } | Select-Object -First 1
    }
    if (-not $preferred) {
        $preferred = $voices | Where-Object { $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'en' } | Select-Object -First 1
    }
    if ($preferred) {
        $synth.SelectVoice($preferred.VoiceInfo.Name)
    }

    $synth.Rate = 0
    $synth.SetOutputToWaveFile($OutputPath)
    $synth.Speak($Text)
}
finally {
    $synth.Dispose()
}
