// --------------------------------------------------------------------
//  Matrix.cpp
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>        // sqrt()
#include <cstdlib>      // RAND_MAX
#include <cstring>      // memset()
#include "Matrix.h"

namespace tfs {     // Tree Frog Software
 
    Matrix::Matrix( unsigned long xx, unsigned long yy, unsigned long zz ):
    m_x( xx ), m_y( yy ), m_z( zz ), m_size( xx * yy * zz ) {
        // Constructor
        if( m_size > 0 ) {
            m_data = new DNN_NUMERIC[m_size];   // Possibly random values.
        }
    }
    
    Matrix::~Matrix( void ) {
        // Destructor
        delete[] m_data;
        m_data = 0;
        m_size = 0;
    }

    unsigned long
    Matrix::width( void ) const {
        return m_x;
    }
    
    unsigned long
    Matrix::height( void ) const {
        return m_y;
    }
    
    unsigned long
    Matrix::depth( void ) const {
        return m_z;
    }
    
    unsigned long
    Matrix::size(   void ) const {
        return m_size;          // = x * w * h;  // Count of DNN_NUMERIC elements.
    }
    
    DNN_NUMERIC*
    Matrix::data( void ) {
        return m_data;          
    }
    
    void
    Matrix::randomize( void ) {
        // Weight normalization is done to equalize the output variance of every neuron,
        // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
        if( m_data != 0 && m_size > 0 ) {
            const DNN_NUMERIC scale = sqrt( 1.0 / m_size );
            DNN_NUMERIC *ptr = m_data;
            const DNN_NUMERIC *end = m_data + m_size;
            while( ptr < end ) {
                *ptr++ = ((DNN_NUMERIC) rand() / (RAND_MAX)) * scale;
            }
        }
        return;
    }
    
    void
    Matrix::zero( void ) {
        if( m_data != 0 && m_size > 0 ) {
            const unsigned long length = m_size * sizeof( DNN_NUMERIC );
            memset( m_data, 0, length );    // Yields IEEE 0 for both integer and real valued variables.
        }
        return;
    }

    
}   // namespace tfs
