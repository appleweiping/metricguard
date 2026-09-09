# Fixed local protocol helper. Caller data is read from stdin, never evaluated.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$stage = 'initialize'
$bitmap = $null
$converted = $null
$memory = $null
$writer = $null
try {
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
    $null = [Windows.Globalization.Language, Windows.Foundation, ContentType=WindowsRuntime]
    $null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
    $null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType=WindowsRuntime]
    $null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Foundation, ContentType=WindowsRuntime]
    $null = [Windows.Storage.Streams.DataWriter, Windows.Foundation, ContentType=WindowsRuntime]
    $awaitMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
        $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    function Await-Operation($operation, [Type] $resultType) {
        $task = $awaitMethod.MakeGenericMethod($resultType).Invoke($null, @($operation))
        return $task.GetAwaiter().GetResult()
    }
    $stage = 'request'
    $buffer = New-Object char[] 8192
    $builder = [Text.StringBuilder]::new()
    while (($count = [Console]::In.Read($buffer, 0, $buffer.Length)) -gt 0) {
        if ($builder.Length + $count -gt 67108864) { throw 'request_limit' }
        $null = $builder.Append($buffer, 0, $count)
    }
    $request = $builder.ToString() | ConvertFrom-Json
    $languages = @([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object {
        $_.LanguageTag
    } | Sort-Object)
    $identity = @{
        os_version = [Environment]::OSVersion.Version.ToString()
        languages = $languages
        max_dimension = [Windows.Media.Ocr.OcrEngine]::MaxImageDimension
    }
    if ($request.action -eq 'capabilities') {
        $response = @{ protocol = 'metricguard.windows-ocr.v1'; action = 'capabilities'; identity = $identity }
    } elseif ($request.action -eq 'recognize') {
        $stage = 'language'
        if ($request.language -isnot [string] -or $languages -cnotcontains $request.language) {
            throw 'language_unavailable'
        }
        $language = [Windows.Globalization.Language]::new($request.language)
        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
        if ($null -eq $engine) { throw 'language_unavailable' }
        $stage = 'decode'
        $bytes = [Convert]::FromBase64String($request.png)
        if ($bytes.Length -gt 33554432) { throw 'image_byte_limit' }
        $memory = [Windows.Storage.Streams.InMemoryRandomAccessStream]::new()
        $writer = [Windows.Storage.Streams.DataWriter]::new($memory)
        $writer.WriteBytes($bytes)
        $null = Await-Operation $writer.StoreAsync() ([uint32])
        $null = $writer.DetachStream()
        $writer.Dispose()
        $writer = $null
        $memory.Seek(0)
        $decoder = Await-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($memory)) ([Windows.Graphics.Imaging.BitmapDecoder])
        if ($decoder.PixelWidth -ne $request.width -or $decoder.PixelHeight -ne $request.height -or
            $decoder.PixelWidth -gt $identity.max_dimension -or $decoder.PixelHeight -gt $identity.max_dimension -or
            [long]$decoder.PixelWidth * [long]$decoder.PixelHeight -gt 20000000) { throw 'image_pixel_limit' }
        $bitmap = Await-Operation $decoder.GetSoftwareBitmapAsync() ([Windows.Graphics.Imaging.SoftwareBitmap])
        $converted = [Windows.Graphics.Imaging.SoftwareBitmap]::Convert(
            $bitmap, [Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8,
            [Windows.Graphics.Imaging.BitmapAlphaMode]::Premultiplied)
        $stage = 'recognize'
        $result = Await-Operation $engine.RecognizeAsync($converted) ([Windows.Media.Ocr.OcrResult])
        $lines = @($result.Lines | ForEach-Object {
            $line = $_
            $words = @($line.Words | ForEach-Object {
                @{ text = $_.Text; box = @($_.BoundingRect.X, $_.BoundingRect.Y, $_.BoundingRect.Width, $_.BoundingRect.Height) }
            })
            @{ text = $line.Text; words = $words }
        })
        $response = @{
            protocol = 'metricguard.windows-ocr.v1'; action = 'recognize'; identity = $identity
            language = $engine.RecognizerLanguage.LanguageTag; lines = $lines; text_angle = $result.TextAngle
        }
    } else { throw 'unknown_action' }
    [Console]::Out.Write(($response | ConvertTo-Json -Depth 10 -Compress))
} catch {
    # Do not print exception messages: decoders can include caller data in them.
    [Console]::Error.Write("native_ocr_failed:$stage")
    exit 2
} finally {
    if ($null -ne $converted) { $converted.Dispose() }
    if ($null -ne $bitmap) { $bitmap.Dispose() }
    if ($null -ne $writer) { $writer.Dispose() }
    if ($null -ne $memory) { $memory.Dispose() }
}
