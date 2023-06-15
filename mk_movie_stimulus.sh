#command to cut segement 0
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=35 out=22585 -video-track  meta.media.width=1280 meta.media.height=720 transition.geometry=1245/690:50x30:100 -transition composite -consumer avformat:test.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 1
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=22185 out=32348 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=36385 out=48273 -video-track  meta.media.width=1280 meta.media.height=720 -transition composite -consumer avformat:test.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 2
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=47873 out=57835 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=58507 out=70446 -video-track  meta.media.width=1280 meta.media.height=720 -transition composite -consumer avformat:seg2.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 3
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=70046 out=86036 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=89332 out=97742 -video-track  meta.media.width=1280 meta.media.height=720 -transition composite -consumer avformat:seg3.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 4
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=97342 out=117391 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=120656 out=123708 -video-track  meta.media.width=1280 meta.media.height=720 -transition composite -consumer avformat:seg4.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 5
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=123308 out=141496 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=145908 out=149671 -video-track  meta.media.width=1280 meta.media.height=720 -transition composite -consumer avformat:seg5.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 6
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=149271 out=152304 -mix 25 -mixer luma forrest_gump_bluray_orig.mkv force_fps=25.000 in=154288 out=178356 -video-track  meta.media.width=1280 meta.media.height=720 -transition composite -consumer avformat:seg6.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k

#command to cut segement 7
melt -profile hdv_720_25p -video-track -attach-track watermark:gray_bars_hdv720.png gray_screen_hdv720.png out=25 forrest_gump_bluray_orig.mkv force_fps=25.000 in=177956 out=194832 -mix 25 -mixer luma -video-track  meta.media.width=1280 meta.media.height=720 -transition composite  -consumer avformat:seg7.mkv f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=5000k