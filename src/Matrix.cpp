// --------------------------------------------------------------------
//  Matrix.cpp
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include <cmath>        // sqrt()
#include <cstring>      // memset()
#include "Error.h"
#include "Matrix.h"
#include "Utility.h"

namespace tfs {     // Tree Frog Software
 
    Matrix::Matrix( const unsigned long xx, const unsigned long yy, const unsigned long zz ):
    m_x( xx ), m_y( yy ), m_z( zz ),
    m_size( xx * yy * zz ),
    m_data( 0 ), m_end( 0 ) {
        // Constructor
        if( m_size > 0 ) {
            m_data = new DNN_NUMERIC[m_size];   // Possibly random values.
            m_end  = m_data + m_size * sizeof( DNN_NUMERIC );
        }
    }
    
    Matrix::~Matrix( void ) {
        // Destructor
        delete[] m_data;
        m_data = 0;
        m_end  = 0;
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
    
    const DNN_NUMERIC*
    Matrix::end( void ) const {
        return m_end;
    }
    
    const DNN_NUMERIC*
    Matrix::dataReadOnly( void ) const {
        return m_data;
    }
    
    void
    Matrix::randomize( void ) {
        // Weight normalization is done to equalize the output variance of every neuron,
        // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
        if( m_data != 0 && m_size > 0 ) {
            const DNN_NUMERIC scale = sqrt( 1.0 / m_size );
                  DNN_NUMERIC *ptr = m_data;
            const DNN_NUMERIC *end = m_end;
            while( ptr < end ) {
                *ptr++ = random( scale );
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
    
    DNN_NUMERIC
    Matrix::dot( const Matrix &matrix ) const {   // Compute the dot product: scalar = lhs (dot) rhs;
        if( matrix.m_size != m_size ) {
            log_error( "Matricies not the same size" );
            return 0.0;
        }
        const DNN_NUMERIC *lhs = m_data;
        const DNN_NUMERIC *end = m_end;
        const DNN_NUMERIC *rhs = matrix.m_data;
              DNN_NUMERIC result = 0.0;
        while( lhs < end ) {
            result += *lhs++ * *rhs++;
        }
        return result;
    }

    
}   // namespace tfs
