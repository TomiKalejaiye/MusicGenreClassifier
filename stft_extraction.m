% ===================================================================
% MATLAB Script To Caclulate STFTs From GTZAN Dataset
% (c) 2019 bjy26@cornell.edu, ok93@cornell.edu, 
% ===================================================================

%List of the genres in our dataset.
genres = {'rock','reggae','pop','metal','jazz','hiphop','disco','country','classical','blues'};

%This outer loop iterates through the genre folders.
epsilon = 0.000001;
for genre = genres
    files = dir(strcat('gtzam/',genre{1},'/*.wav'));
    numel = size(files);
    %This loop iterates through each song in each genre.
    for i = 1:numel(1)
        fprintf(1,'Getting ST-DFT for %s\n',files(i).name)
        [s,F_s] = audioread(strcat('gtzam/',genre{1},'/',files(i).name));
        spec_window_sz = 33090;
        s = s(:,1);
        %As we are breaking each song into 1.5s windows, this loop iterates
        %through those windows.
        for i_s = 1:spec_window_sz/2:length(s)-spec_window_sz
            sub_s = s(i_s:i_s+spec_window_sz);
            T_s = 1/F_s;

            N = 1024;

            L = length(sub_s);

            n = (0:1:L-1).';

            m = 1:N/4:L-N;

            w = sin(pi*(n(1:N)+0.5)/N).^2;

            k = (0:1:N-1).';

            STDFT = zeros(length(m),N);
            
            %Calculating our STDFTs
            for idm=1:length(m)
                s_w = sub_s((idm-1)*N/4+(1:N)).*w;
                phase = exp((-1i*2*pi*k*(m(idm)-1))/N);
                STDFT(idm,:) = phase.*fft(s_w,N)./sqrt(N);
            end
            
            %Only take the positive frequencies, and convert to decibels
            samples = 1:length(m);
            frequency = 1:N/2;
            magnitude = (20*log10(abs(STDFT(samples,frequency))+epsilon));
            
%            figure(2);
%
%             imagesc(magnitude(samples,frequency).')
%             set(gca,'YDir','normal')
%             
%             title('Spectrogram Of Hip Hop Song');
% 
%             cb = colorbar;
%             title(cb,'Amplitude (dB)');
%             xlabel('Samples');
%             ylabel('Frequency (Hz)'); 
% 
%             yticks([0 3000 6000 9000 12000 15000 18000 21000] .* (T_s*N));
% 
%             yticklabels(string([0 3000 6000 9000 12000 15000 18000 21000]));
%             
%             saveas(gcf,strcat(genre{1},int2str(i),int2str(i_s),'.png'));
%             break

            %Save out output STFTs.
            save(strcat('stdft_data_gtzam/',genre{1},int2str(i),int2str(i_s),'.mat'),'magnitude')
        end
    end
end
