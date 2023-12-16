%% Input data , set parameters
clear all,
close all ;
Ii =imread('1.bmp');% 大小为 256 x256 pix的8位位图
Ii = double(Ii);
PH = rand([256,256]);
Ii=Ii.*exp(2i*pi*PH);% 在物体条纹上叠加随机相位
% 向前传播到全息图
M =256;
z=30;%(单位:cm,物体和观察者之间的距离)
w=6710*10^-8;%(单位:cm，波长)671nm
delta=0.0050;%单位:cm,像素间距为8微米
r=1:M;
c=1:M;
[C,R] = meshgrid(c,r);
p=exp(-2i*pi*z.*((1/w)^2-(1/M/delta)^2.*(C-M/2-1).^2-...
(1/M/delta)^2.*(R-M/2-1).^2).^0.5);
A0 = fftshift(fft2(fftshift(Ii))) ;
Az=A0.*p;
E = ifftshift(ifft2(ifftshift(Az))) ;
ES = zeros(512);
ES(200:265,128:383) = E(72:137,1:256);% 狭缝状孔径
figure;imshow(E);title( 'Fresnel_holo images')
imwrite(real(E), 'Fresnel_holo.bmp')
figure;imshow(ES);title( 'H1 images')
imwrite(real(ES), 'H1.bmp')
% 2.向前传播到全息图2
M=512;
r=1:M;
c=1:M;
[C,R] = meshgrid(c,r);
A1 = fftshift(fft2(fftshift(conj(ES))));
z=60;%全息图1和全息图2间距离
p=exp(-2i*pi*z.*((1/w)^2-(1/M/delta)^2.*(C-M/2-1).^2-...
(1/M/delta)^2.*(R-M/2-1).^2).^0.5);
Az1=A1.*p;
H=ifftshift(ifft2(ifftshift(Az1)));% 全息面上的物光
%h = (abs(H.*conj(H)));%振幅型
h = real(H);%振幅型
%h = exp(1i*angle(H));%相位型
h = h/max(max(h));
figure;imshow(h);title( 'H2 images')
%imwrite(real(h), 'H2_pha.bmp')%相位型
imwrite(h, 'H2_amp.bmp')%振幅
%3.重构(650nm)
H2 = zeros(1024);
H2(256:767,256:767) = conj(h) ;

M =1024;
r=1:M;
c=1:M;
[C,R] = meshgrid(c,r);
z=30;
p=exp(-2i*pi*z.*((1/w)^2-(1/M/delta)^2.*(C-M/2-1).^2-...
(1/M/delta)^2.*(R-M/2-1).^2).^0.5);
A2 = fftshift(fft2(fftshift(H2)));
Az2=A2.*p;
R650 = ifftshift(ifft2(ifftshift(Az2))) ;
R650 =(abs( R650)).^2;
R650 = R650/max(max(R650));
%R650 = R650(384:639,384:639) ;
figure;imshow(R650);
title( 'Reconstructed image(650nm)')
axis off

Az2=Az2.*p;%观察平面
S650 = ifftshift(ifft2(ifftshift(Az2)));
S650 =(abs(S650)).^2;
S650 = S650/max(max(S650));
%S650 = S650(384:639,384:639);
%重构(550nm)
w2=5500*10^-8;
H3 =H2.*exp(2i*pi*sind(0.4)*(w-w2)/w/w2.*R*delta);
p=exp(-2i*pi*z.*((1/w2)^2-(1/M/delta)^2.*(C-M/2-1).^2-...
(1/M/delta)^2.*(R-M/2-1).^2).^0.5);
Az3 = fftshift(fft2(fftshift( H3 ))).* p;
R550 = ifftshift(ifft2(ifftshift( Az3 )) ) ;
R550 =(abs(R550)).^2;
R550= R550/max(max(R550));
%R550 = R550(384:639 ,384:639) ;
figure; imshow( R550);
title( 'Reconstructed image ( 550nm )')
Az3=Az3.*p;
S550 = ifftshift( ifft2 ( ifftshift( Az3 ) )); 
S550 = (abs( S550)).*2;
S550= S550/max(max(S550));
%S550 = S550(384:639,384:639); 
%重构(450nm)
w3 =4500*10^-8;
H4 = H2.*exp(2i*pi*sind(0.4)* (w-w3)/w/w3.*R* delta);

p=exp(-2i*pi*z.*((1/w3)^2-(1/M/delta)^2.*(C-M/2-1)^.2-...
(1/M/delta)^2.*(R-M/2-1).^2).^0.5);
Az4 = fftshift( fft2 ( fftshift(H4))).*p;
R450 = ifftshift( ifft2( ifftshift( Az4) ) ) ;
R450 =(abs( R450)).^2;
R450 = R450/max( max( R450)) ;
%R450 = R450(384 :639 ,384 :639);
figure ; imshow(R450);
title( 'reconstructed image ( 450nm)')

A4=Az4.*p;
S450 = ifftshift(ifft2(ifftshift( Az4 )));
S450=(abs(S450)).^2;
S450 = S450/max( max( S450));
%S450 = S450(384:639,384:639);
% 彩色狭缝像
SLIT = zeros(1024,1024,3) ;
SLIT( :,:,1) = S650;
SLIT(:,:,2) = S550;
SLIT( :,:,3) = S450;
SLIT = uint8(SLIT.*500);
figure ; 
image(SLIT)
imwrite(SLIT, 'rec_amp.bmp')
title( 'Slit images')
axis off
axis equal
