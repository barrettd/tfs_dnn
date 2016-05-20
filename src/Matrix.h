// --------------------------------------------------------------------
//  Matrix.hpp - 3D matrix <template> for DNN_NUMERIC and DNN_INTEGER
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#ifndef Matrix_h
#define Matrix_h

#include <cmath>        // sqrt()
#include <cstring>      // memset()
#include "Constants.h"
#include "Error.h"
#include "Utility.h"    // random()

namespace tfs {     // Tree Frog Software

    template <typename T> class TMatrix  {      // Possibly use Eigen matricies: https://eigen.tuxfamily.org/
    protected:
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_size;   // = x * w * h;  // Count of DNN_NUMERIC elements.
        T            *m_data;
        T            *m_end;
        
        void allocate( void ) {
            if( m_size > 0 ) {
                m_data = new T[m_size];         // Possibly random values.
                m_end  = m_data + m_size * sizeof( T );
            }
        }

    public:
        TMatrix( const TMatrix &other ) :       // Constructor
        m_x( other.m_x ), m_y( other.m_y ), m_z( other.m_z ), m_size( other.m_size ),
        m_data( 0 ), m_end( 0 ) {
            allocate();
        }

        TMatrix( const unsigned long xx, const unsigned long yy = 1, const unsigned long zz = 1 ):  // Constructor
        m_x( xx ), m_y( yy ), m_z( zz ), m_size( xx * yy * zz ),
        m_data( 0 ), m_end( 0 ) {
            allocate();
        }
        
        ~TMatrix( void ) {      // Destructor
            delete[] m_data;
            m_data = 0;
            m_end  = 0;
            m_size = 0;
        }
        
        inline unsigned long width(  void ) const { return m_x; }
        inline unsigned long height( void ) const { return m_y; }
        inline unsigned long depth(  void ) const { return m_z; }
        inline unsigned long size(   void ) const { return m_size; }    // Count of elements.
        
        inline bool isEmpty( void ) const { return m_data == 0 || m_size < 1; }
        
        inline       T* data( void ) {               return m_data; }   // Start of data array
        inline const T* end(  void ) const {         return m_end;  }   // End of data array
        inline const T* dataReadOnly( void ) const { return m_data; }   // Start of data array
        
        void randomize( void ) {                // Fill matrix with small positive values
            // Weight normalization is done to equalize the output variance of every neuron,
            // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
            if( m_data != 0 && m_size > 0 ) {
                const T scale = sqrt( 1.0 / m_size );
                T *ptr = m_data;
                const T *end = m_end;
                while( ptr < end ) {
                    *ptr++ = random( scale );
                }
            }
        }

        void zero( void ) {                     // Fill matrix with zeros
            if( m_data != 0 && m_size > 0 ) {
                const unsigned long length = m_size * sizeof( T );
                memset( m_data, 0, length );    // Yields IEEE 0 for both integer and real valued variables.
            }
        }
        
        T dot( const TMatrix &matrix ) const {     // Calculate the dot product: scalar = lhs (dot) rhs;
            if( m_data == 0 || m_end == 0 || m_size < 1 ) {
                log_error( "empty matrix" );
                return 0;
            }
            if( matrix.m_size != m_size ) {
                log_error( "Matricies not the same size" );
                return 0;
            }
            const T *       lhs = m_data;
            const T * const end = m_end;
            const T *       rhs = matrix.m_data;
            T result = 0;
            while( lhs < end ) {
                result += *lhs++ * *rhs++;
            }
            return result;
        }
        
        T max( void ) const {                       // Return maximum value.
            if( m_data == 0 || m_end == 0 || m_size < 1 ) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            T       result = *data++;
            while( data < end ) {
                if( *data > result ) {
                    result = *data;
                }
                data++;
            }
            return result;
        }
        
    };
    
    typedef TMatrix< DNN_NUMERIC >  Matrix;     // Typically double
    typedef TMatrix< DNN_INTEGER > DMatrix;     // Typically long
    
}   // namespace tfs

#endif /* Matrix_h */
