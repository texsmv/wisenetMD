#include "wisenetMD.h"

void Erosion( int erosion_elem, int erosion_size, cv::Mat& src){
    int erosion_type = 0;
    if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }
    cv::Mat element = cv::getStructuringElement( erosion_type,
                        cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                        cv::Point( erosion_size, erosion_size ) );
    cv::erode( src, src, element );
    
}
void Dilation(int dilation_elem, int dilation_size, cv::Mat& src){
    int dilation_type = 0;
    if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }
    cv::Mat element = cv::getStructuringElement( dilation_type,
                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        cv::Point( dilation_size, dilation_size ) );
    cv::dilate( src, src, element );
    
}


float calc_ratio(bool* h_mask, int h, int w){
    // int n_back_p = thrust::reduce(thrust::host, h_mask, h_mask + (h * w));
    int n_back_p = 0;
    for (size_t i = 0; i < (h * w); i++) {
      if (h_mask[i])
        n_back_p ++;
    }
  
    return (float) n_back_p / (float)(h * w);
    // return (float) n_back_p ;
}

WisenetMD::WisenetMD()
{
}

WisenetMD::~WisenetMD()
{
}



void WisenetMD::init(){
    std::cout<<"\n------------------------init--------------------------\n";
    /* Creating host arrays*/
    h_B_color = new uchar[height * width * n_B * 3];
    h_B_texture = new int[height * width * n_B];
    h_DB = new uchar[height * width * n_DB];
    h_T = new float[height * width];
    h_R_color = new float[height * width];
    h_R_texture = new float[height * width];
    h_S = new bool[height * width];
    h_S_l = new bool[height * width];
    h_DR = new bool[height * width];
    h_BR = new float[height * width];
    h_TB = new float[height * width];
    h_Dist_last = new float[height * width];
    h_S_feed = new float[height * width];
    h_d = new float[height * width];
    h_D_min = new float[height * width];
    h_v = new float[height * width];
    h_w = new float[height * width];
    h_R = new float[height * width];
    
    


    /* Creating device arrays*/
    d_B_color = cudu::array<uchar>(height * width * n_B * 3);
    d_B_texture = cudu::array<int>(height * width * n_B);
    d_DB = cudu::array<uchar>(height * width * n_DB * 3);
    d_R_color = cudu::array<float>(height * width);
    d_R_texture = cudu::array<float>(height * width);
    d_S = cudu::array<bool>(height * width);
    d_S_l = cudu::array<bool>(height * width);
    d_DR = cudu::array<bool>(height * width);
    d_BR = cudu::array<float>(height * width);
    d_BR = cudu::array<float>(height * width);
    d_T = cudu::array<float>(height * width);
    d_TB = cudu::array<float>(height * width);
    d_color_l = cudu::array<uchar>(height * width * 3);
    d_texture_l = cudu::array<int>(height * width);
    d_Dist_last = cudu::array<float>(height * width);
    d_S_feed = cudu::array<float>(height * width);
    d_d = cudu::array<float>(height * width);
    d_D_min = cudu::array<float>(height * width);
    d_v = cudu::array<float>(height * width);
    d_w = cudu::array<float>(height * width);
    d_R = cudu::array<float>(height * width);

       
    std::cout<<"Creating curand states\n";
    states = cudu::array<curandState_t>(height * width);
    std::cout<<"Initializing curand states\n";
    kernels::init_randoms<<<grid, block>>>(time(0), states, height, width);

    
    devs();
    std::cout<<"\n------------------------------------------------------\n";
}

void WisenetMD::init_model(){
    std::cout<<"\n------------------------init model--------------------------\n";
    block = dim3(16, 16, 1);
    grid = dim3(ceil( height/ float(block.x)), ceil(width / float(block.y)));

    std::cout<<height << " "<<width<<std::endl;
    

    
    /* Initializing R */
    cudu::fill<float>(d_R_color, Ro_color, height * width);
    cudu::fill<float>(d_R_texture, Ro_texture, height * width);
    /* Initializing B */
    kernels::init_B_samples<<<grid, block>>>(d_B_color, d_B_texture, n_B, input->d_rgb, input->d_texture, height, width);
    /* Initializing DB */
    kernels::init_DB_samples<<<grid, block>>>(d_DB, n_DB, input->d_rgb, height, width);
    devs();
    /* Initializing d_D_min */
    cudu::fill<float>(d_D_min, 0.5, height * width);
    /* Initializing TB */
    cudu::fill<float>(d_TB, 0.0, height * width);
    /* Initializing d_R */
    cudu::fill<float>(d_R, 2, height * width);
    /* Initializing d_S_l */
    cudu::fill<bool>(d_S_l, 1, height * width);
    /* Initializing d_BR */
    cudu::fill<float>(d_BR, 0, height * width);
    /* Initializing T */
    cudu::fill<float>(d_T, 0.5, height * width);
    /* Initializing v */
    cudu::fill<float>(d_v, 1, height * width);
    /* Initializing S_feed */
    cudu::fill<float>(d_S_feed, 0.4, height * width);
    
    

    /* windows */
    m_S = cv::Mat(height, width, CV_8UC3); /* initializa mat S*/
    m_DR = cv::Mat(height, width, CV_8UC3); /* initializa mat DR*/
    m_T = cv::Mat(height, width, CV_8UC3); /* initializa mat T*/
    m_R = cv::Mat(height, width, CV_8UC3); /* initializa mat R*/
    m_BR = cv::Mat(height, width, CV_8UC3); /* initializa mat BR*/
    m_v = cv::Mat(height, width, CV_8UC3); /* initializa mat v*/
    m_d = cv::Mat(height, width, CV_8UC3); /* initializa mat d*/
    m_D_min = cv::Mat(height, width, CV_8UC3); /* initializa mat D_min*/
    m_Dist_last = cv::Mat(height, width, CV_8UC3); /* initializa mat Dist last*/
    m_S_feed = cv::Mat(height, width, CV_8UC3); /* initializa mat S_feed*/
    m_w = cv::Mat(height, width, CV_8UC3); /* initializa mat w*/
    m_R_color = cv::Mat(height, width, CV_8UC3); /* initializa mat R_color*/
    m_R_texture = cv::Mat(height, width, CV_8UC3); /* initializa mat R_texture*/
    m_TB = cv::Mat(height, width, CV_8UC3); /* initializa mat TB*/

    cv::namedWindow("S");
    cv::moveWindow("S", 10, 100);
    cv::namedWindow("R");
    cv::moveWindow("R", 200, 100);
    cv::namedWindow("T");
    cv::moveWindow("T", 400, 100);
    cv::namedWindow("DR");
    cv::moveWindow("DR", 800, 100);
    cv::namedWindow("BR");
    cv::moveWindow("BR", 1000, 100);
    cv::namedWindow("v");
    cv::moveWindow("v", 10, 600);
    cv::namedWindow("D_min");
    cv::moveWindow("D_min", 200, 600);
    cv::namedWindow("d");
    cv::moveWindow("d", 400, 600);
    cv::namedWindow("S_feed");
    cv::moveWindow("S_feed", 600, 600);
    cv::namedWindow("Dist_last");
    cv::moveWindow("Dist_last", 800, 600);
    cv::namedWindow("w");
    cv::moveWindow("w", 1000, 600);
    cv::namedWindow("R_color");
    cv::moveWindow("R_color", 800, 900);
    cv::namedWindow("R_texture");
    cv::moveWindow("R_texture", 200, 900);
    cv::namedWindow("TB");
    cv::moveWindow("TB", 200, 400);
    
    
    devs();
    std::cout<<"\n------------------------------------------------------------\n";
}


/* -------------------------------- multiple operation----------------------------*/

void WisenetMD::process_frame(){
    t++;
    /* show mats*/
    devs();
    cudu::d2h<bool>(d_S, h_S, height * width);
    cudu::d2h<bool>(d_DR, h_DR, height * width);
    cudu::d2h<float>(d_R, h_R, height * width);
    cudu::d2h<float>(d_BR, h_BR, height * width);
    cudu::d2h<float>(d_T, h_T, height * width);
    cudu::d2h<float>(d_v, h_v, height * width);
    cudu::d2h<float>(d_d, h_d, height * width);
    cudu::d2h<float>(d_D_min, h_D_min, height * width);
    cudu::d2h<float>(d_w, h_w, height * width);
    cudu::d2h<float>(d_S_feed, h_S_feed, height * width);
    cudu::d2h<float>(d_Dist_last, h_Dist_last, height * width);
    cudu::d2h<float>(d_R_color, h_R_color, height * width);
    cudu::d2h<float>(d_R_texture, h_R_texture, height * width);
    cudu::d2h<float>(d_TB, h_TB, height * width);
    
    devs();
    std::cout<<"S: ";
    cvu::T2mat<bool>(m_S, height, width, h_S, 1);
    std::cout<<"DR: ";
    cvu::T2mat<bool>(m_DR, height, width, h_DR, 1);
    std::cout<<"R: ";
    cvu::T2mat<float>(m_R, height, width, h_R, 4.0);
    std::cout<<"T: ";
    cvu::T2mat<float>(m_T, height, width, h_T, 255.0);
    std::cout<<"BR: ";
    cvu::T2mat<float>(m_BR, height, width, h_BR, 1);
    std::cout<<"TB: ";
    cvu::T2mat<float>(m_TB, height, width, h_TB, 100);
    std::cout<<"v: ";
    cvu::T2mat<float>(m_v, height, width, h_v, 1.0);
    std::cout<<"d: ";
    cvu::T2mat<float>(m_d, height, width, h_d, 1.0);
    std::cout<<"D_min: ";
    cvu::T2mat<float>(m_D_min, height, width, h_D_min, 1.0);
    std::cout<<"w: ";
    cvu::T2mat<float>(m_w, height, width, h_w, 1.5);
    std::cout<<"Dist_last: ";
    cvu::T2mat<float>(m_Dist_last, height, width, h_Dist_last, 1.0);
    std::cout<<"S_feed: ";
    cvu::T2mat<float>(m_S_feed, height, width, h_S_feed, 1.0);
    std::cout<<"R_color: ";
    cvu::T2mat<float>(m_R_color, height, width, h_R_color, 255.0);
    std::cout<<"R_texture: ";
    cvu::T2mat<float>(m_R_texture, height, width, h_R_texture, 25.0);


    devs();


    Erosion(2, 1, m_S);
    Dilation(2, 1, m_S);

    ratio_movement = calc_ratio(h_S, height, width);
    

    cv::imshow("BR", m_BR);
    cv::imshow("DR", m_DR);
    cv::imshow("S", m_S);
    cv::imshow("R", m_R);
    cv::imshow("T", m_T);
    cv::imshow("v", m_v);
    cv::imshow("d", m_d);
    cv::imshow("D_min", m_D_min);
    cv::imshow("w", m_w);
    cv::imshow("S_feed", m_S_feed);
    cv::imshow("Dist_last", m_Dist_last);
    cv::imshow("R_texture", m_R_texture);
    cv::imshow("R_color", m_R_color);
    cv::imshow("TB", m_TB);



    /*calculate d_R_color/texture*/
    kernels::calc_R_color_texture<<<grid,block>>>(d_R, d_R_color, d_R_texture, Ro_color, Ro_texture, height, width);

    
    // /* calculate mask*/
    calc_S();
    devs();

    // /* update samples */
    update_B();
    devs();
    

    /* FP re-check */
    kernels::calc_DR<<<grid, block>>>(d_S, d_S_l, d_DR, d_BR, d_TB, t, blink_threshold, height, width); 
    devs();
    kernels::FP_recheck<<<grid, block>>>(d_DB, n_DB, d_DR, d_S, input->d_rgb, DB_color_threshold, height, width);
    devs();
    kernels::calc_Dist_last<<<grid, block>>>(d_Dist_last, input->d_rgb, input->d_texture, d_color_l, d_texture_l, height, width);
    devs();
    kernels::calc_S_feed<<<grid, block>>>(d_S_feed, d_S_l, 0.1, height, width);
    devs();
    kernels::update_DB<<<grid, block>>>(d_DB, n_DB, d_DR, d_S, d_S_l, d_Dist_last, d_S_feed, input->d_rgb, height, width, states);
    devs();
    
    /* Update T and R*/
    kernels::calc_d<<<grid, block>>>(d_d, d_B_color, d_B_texture, n_B, d_R_color, d_R_texture, input->d_rgb, input->d_texture, height, width);
    devs();
    kernels::calc_D_min<<<grid, block>>>(d_D_min, d_d, alpha, height, width);
    devs();
    kernels::calc_w<<<grid, block>>>(d_w, d_DR, d_Dist_last, d_S_feed, height, width);
    devs();
    kernels::calc_v<<<grid, block>>>(d_v, v_decr, d_S, d_S_l, d_w, height, width);
    devs();
    kernels::update_R<<<grid, block>>>(d_R, d_v, d_D_min, height, width);
    devs();
    kernels::update_T<<<grid, block>>>(d_T, d_S, d_v, d_D_min, height, width);
    devs();

    /* save last values*/
    save_last_values();

    std::cout<<"\n\n";


}

void WisenetMD::update_B(){
    kernels::update_B<<<grid, block>>>(d_S, d_B_color, d_B_texture, n_B, input->d_rgb, input->d_texture, d_T, height, width, states);
}


/* ---------------------------------operations-------------------------------------*/


void WisenetMD::calc_S(){
    kernels::calc_S<<<grid, block>>>(d_S, d_B_color,  d_B_texture,  n_B, d_R_color, d_R_texture, input->d_rgb, input->d_texture, height, width);
}

void WisenetMD::save_last_values(){
    cudu::d2d<bool>(d_S, d_S_l, height * width);
    cudu::d2d<uchar>(input->d_rgb, d_color_l, height * width * 3);
    cudu::d2d<int>(input->d_texture, d_texture_l, height * width);
    devs();
}


/*-------------------------------- getters and setters ----------------------------*/
void WisenetMD::set_input(VideoInput* input){
    this->input = input;
    this->width = input->get_size().width;
    this->height = input->get_size().height;
}

float WisenetMD::get_movement_ratio(){
    return ratio_movement;
}