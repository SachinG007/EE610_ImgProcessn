function varargout = assgn2(varargin)
% ASSGN2 MATLAB code for assgn2.fig
%      ASSGN2, by itself, creates a new ASSGN2 or raises the existing
%      singleton*.
%
%      H = ASSGN2 returns the handle to a new ASSGN2 or the handle to
%      the existing singleton*.
%
%      ASSGN2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ASSGN2.M with the given input arguments.
%
%      ASSGN2('Property','Value',...) creates a new ASSGN2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before assgn2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to assgn2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help assgn2

% Last Modified by GUIDE v2.5 06-Oct-2018 12:08:53

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @assgn2_OpeningFcn, ...
                   'gui_OutputFcn',  @assgn2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before assgn2 is made visible.
function assgn2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to assgn2 (see VARARGIN)

% Choose default command line output for assgn2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes assgn2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = assgn2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA
global image imR imG imB colourimg image_copy m n im_prev deblurred
[path,cancel]=imgetfile();
if cancel
    msgbox(sprintf('Error'),'Error','Error');
    return
else
    image = imread(path);%The image variable stores the input image as a matrix
    image = im2double(image);
    image_copy=image;
    im_prev = image;
    deblurred = image;
    %Following is to figure out if the input image is a black and white image or a colour image
    [m, n, numberOfColorChannels] = size(image);
    if numberOfColorChannels==1
        colourimg=0;
    elseif numberOfColorChannels==3
        colourimg=1;
        imR=image(:,:,1);
        imG=image(:,:,2);
        imB=image(:,:,3);
    end
    axes(handles.axes1);
    imshow(image);
    axes(handles.axes2);
    imshow(deblurred);
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image imR imG imB colourimg m n im_prev deblurred
load k11;
h=im2double(k11);
[~,n1]=size(h);
h=h/(sum(sum(h(:,:))));
n1=(n1-1)/2;
im_blur=image;
limit1=m;
limit2=n;
if colourimg ~= 1
    for i=1:limit1
        for j=1:limit2
            if(i>n1+1 && j>n1+1 && i<limit1-n1 && j<limit2-n1)
                im_blur(i,j)=sum(sum(image((i-n1):(i+n1),(j-n1):(j+n1)).*h));
            else
                add=0;
                for a=-n1:n1
                    for b=-n1:n1
                        if ((i+a)>=1 && (i+a)<=limit1 && (j+b)>=1 && (j+b)<=limit2)
                            add=add+image(i+a,j+b)*h(a+n1+1,b+n1+1);
                        end
                    end
                end
                im_blur(i,j)=add;
            end
        end
    end
else
    for i=1:limit1
        for j=1:limit2
            if(i>n1+1 && j>n1+1 && i<limit1-n1 && j<limit2-n1)
                im_blur(i,j,1)=sum(sum(image((i-n1):(i+n1),(j-n1):(j+n1),1).*h));
                im_blur(i,j,2)=sum(sum(image((i-n1):(i+n1),(j-n1):(j+n1),2).*h));
                im_blur(i,j,3)=sum(sum(image((i-n1):(i+n1),(j-n1):(j+n1),3).*h));
            else
                add=zeros(1,3);
                for a=-n1:n1
                    for b=-n1:n1
                        if ((i+a)>=1 && (i+a)<=limit1 && (j+b)>=1 && (j+b)<=limit2)
                            add(1,1)=add(1,1)+image(i+a,j+b,1)*h(a+n1+1,b+n1+1);
                            add(1,2)=add(1,2)+image(i+a,j+b,2)*h(a+n1+1,b+n1+1);
                            add(1,3)=add(1,3)+image(i+a,j+b,3)*h(a+n1+1,b+n1+1);
                        end
                    end
                end
                im_blur(i,j,:)=add;
            end
        end
    end
end
axes(handles.axes2);
imshow(im_blur);
im_prev = deblurred;
deblurred = im_blur;
display('Done');

% --- Executes on button press in inverse_k1.
function inverse_k1_Callback(hObject, eventdata, handles)
% hObject    handle to inverse_k1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image imR imG imB colourimg image_copy m n deblurred
R = 20;
R_sq = R^2;
load k11;
h=k11;
h=im2double(h);
h=rot90(h,2);
h=h/255.0;
[~,h_ffts] = my_fft2(h,2*m,2*n);
if colourimg == 1
    [~,R_ffts] = my_fft2(imR,2*m,2*n);
    gR_ffts=zeros(2*m,2*n);
    for i=1:2*m
        for j=1:2*n
            if (i^2+j^2)<R_sq
                gR_ffts(i,j)=R_ffts(i,j)/h_ffts(i,j);
            else
                gR_ffts(i,j)=R_ffts(i,j);
            end
        end
    end
    gR_fft = fftshift(gR_ffts);
    gR = ifft2(gR_fft);
    gR = gR(1:m,1:n);
    
    [~,G_ffts] = my_fft2(imG,2*m,2*n);
    gG_ffts=zeros(2*m,2*n);
    for i=1:2*m
        for j=1:2*n
            if (i^2+j^2)<R_sq
                gG_ffts(i,j)=G_ffts(i,j)/h_ffts(i,j);
            else
                gG_ffts(i,j)=G_ffts(i,j);
            end
        end
    end
    gG_fft = fftshift(gG_ffts);
    gG = ifft2(gG_fft);
    gG = gG(1:m,1:n);
    
    [~,B_ffts] = my_fft2(imB,2*m,2*n);
    gB_ffts=zeros(2*m,2*n);
    for i=1:2*m
        for j=1:2*n
            if (i^2+j^2)<R_sq
                gB_ffts(i,j)=B_ffts(i,j)/h_ffts(i,j);
            else
                gB_ffts(i,j)=B_ffts(i,j);
            end
        end
    end
    gB_fft = fftshift(gB_ffts);
    gB = ifft2(gB_fft);
    gB = gB(1:m,1:n);
    
    g = zeros(m,n,3);
    g(:,:,1) = gR;
    g(:,:,2) = gG;
    g(:,:,3) = gB;
    deblurred = g;
    axes(handles.axes2);
    imshow(deblurred);
    display('Done');
else
    %calculation of the fft
    
    [~,image_ffts] = my_fft2(image,2*m,2*n);
    gIm_ffts=zeros(2*m,2*n);
    for i=1:2*m
        for j=1:2*n
            if (i^2+j^2)<R_sq
                gIm_ffts(i,j)=image_ffts(i,j)/h_ffts(i,j);
            else
                gIm_ffts(i,j)=image_ffts(i,j);
            end
        end
    end
    gIm_fft = fftshift(gIm_ffts);
    gIm = ifft2(gIm_fft);
    gIm = gIm(1:m,1:n);
    deblurred = gIm;
    axes(handles.axes2);
    imshow(deblurred);
end
display('done deblurring');

% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global deblurred
im_save = deblurred;
axes(handles.axes3);
imshow(im_save);
imwrite(im_save,'ImageSavedya.jpg','jpg','Comment','My JPEG file')

% --- Executes on button press in reset.
function reset_Callback(hObject, eventdata, handles)
% hObject    handle to reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global image image_copy deblurred
image = image_copy;
deblurred = image;
axes(handles.axes2);
imshow(deblurred);